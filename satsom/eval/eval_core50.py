import logging
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd

from satsom.model import SatSOM, SatSOMParameters
from satsom.eval.knn import KNNClassifier
from satsom.eval.dsdm import DSDMClassifier
from satsom.eval.propre import PROPREClassifier


# ---------------------------------------------------------
# 1. Feature Extractor & Dataset Logic
# ---------------------------------------------------------
class Core50SessionDataset(Dataset):
    """
    Custom loader for Core50 that tracks Session IDs.
    Structure assumed:
       root/
         s1/
           o1/ -> images
           o2/ ...
         s2/
           ...
    """

    def __init__(self, root, transform=None):
        self.samples = []  # (path, label, session_id)
        self.transform = transform

        if not os.path.exists(root):
            raise FileNotFoundError(f"Core50 root not found: {root}")

        # 1. Scan for all unique Object names to build class index
        object_names = set()

        # Valid sessions in Core50 are s1..s11.
        # We scan everything but only parse those matching 'sX'.
        for session_name in os.listdir(root):
            if not session_name.startswith("s"):
                continue

            session_path = os.path.join(root, session_name)
            if not os.path.isdir(session_path):
                continue

            for obj_name in os.listdir(session_path):
                # We assume folders are named 'o{number}'
                if obj_name.startswith("o") and os.path.isdir(
                    os.path.join(session_path, obj_name)
                ):
                    object_names.add(obj_name)

        # Sort objects naturally (o1...o50)
        try:
            sorted_objects = sorted(
                list(object_names), key=lambda x: int(x.replace("o", ""))
            )
        except ValueError:
            sorted_objects = sorted(list(object_names))

        self.class_to_idx = {o: i for i, o in enumerate(sorted_objects)}
        print(f"Found {len(self.class_to_idx)} unique classes.")

        # 2. Collect all images with Session ID
        for session_name in os.listdir(root):
            if not session_name.startswith("s"):
                continue

            # Extract session number (e.g. 's1' -> 1)
            try:
                session_id = int(session_name.replace("s", ""))
            except ValueError:
                continue

            session_path = os.path.join(root, session_name)
            if not os.path.isdir(session_path):
                continue

            for obj_name in os.listdir(session_path):
                if obj_name not in self.class_to_idx:
                    continue

                obj_path = os.path.join(session_path, obj_name)
                lbl = self.class_to_idx[obj_name]

                for fname in os.listdir(obj_path):
                    if fname.lower().endswith(("png", "jpg", "jpeg")):
                        self.samples.append(
                            (os.path.join(obj_path, fname), lbl, session_id)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl, sid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Return tuple: (image, label, session_id)
        return img, lbl, sid


def get_core50_data(
    root: str, device: str, batch_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts ResNet features and splits them into Train/Test based on Core50 protocol.

    Protocol:
    - Test Sessions: 3, 7, 10
    - Train Sessions: 1, 2, 4, 5, 6, 8, 9, 11

    Returns: (train_x, train_y, test_x, test_y)
    """
    print(f"Generating embeddings using ResNet18 on {device}...")

    # Standard ImageNet normalization
    transform = transforms.Compose(
        [
            transforms.Resize(
                (128, 128)
            ),  # Core50 native size is 128, but ResNet likes 224 usually.
            # However, for speed on Core50, 128 is often used with adaptive pooling.
            # If you strictly want ResNet stats, resize to 224.
            # We will resize to 224 to be safe for the pre-trained weights.
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    dataset = Core50SessionDataset(root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_feats = []
    all_labels = []
    all_sessions = []

    with torch.no_grad():
        for imgs, lbls, sids in tqdm(loader, desc="Extracting Features"):
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = feats.view(feats.size(0), -1)

            all_feats.append(feats.cpu())
            all_labels.append(lbls)
            all_sessions.append(sids)

    features = torch.cat(all_feats)
    labels = torch.cat(all_labels)
    sessions = torch.cat(all_sessions)

    # Core50 Official Split
    test_mask = (sessions == 3) | (sessions == 7) | (sessions == 10)
    train_mask = ~test_mask

    train_x, train_y = features[train_mask], labels[train_mask]
    test_x, test_y = features[test_mask], labels[test_mask]

    print("Data Split Complete.")
    print(f"Train: {train_x.shape[0]} samples (Sessions 1,2,4,5,6,8,9,11)")
    print(f"Test:  {test_x.shape[0]} samples (Sessions 3,7,10)")

    return train_x, train_y, test_x, test_y


def eval_som(
    som_params: SatSOMParameters,
    core50_root: str,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_model: bool = False,
):
    """
    Executes the Core50 NC (New Classes) Benchmark.

    Setting: Class-Incremental
    Total Classes: 50
    Tasks: 9
       - Task 0: 10 classes
       - Task 1-8: 5 classes each
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging setup
    logger = logging.getLogger("core50_nc")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(ch)

    # 1. Load Data & Extract Features
    logger.info("Preparing Data...")
    train_x, train_y, test_x, test_y = get_core50_data(core50_root, device)

    # Move to GPU
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    # 2. Define Tasks (NC Scenario)
    # Task 0: Classes 0-9 (10 classes)
    # Task 1..8: Classes 10-14, 15-19, ... (5 classes each)
    n_total_classes = 50

    tasks = []

    # First batch: 10 classes
    tasks.append(list(range(0, 10)))

    # Remaining batches: 5 classes each
    curr = 10
    while curr < 50:
        end = min(curr + 5, 50)
        tasks.append(list(range(curr, end)))
        curr = end

    logger.info(f"Defined {len(tasks)} tasks.")
    logger.info(f"Task 0 classes: {tasks[0]}")
    logger.info(f"Task 1 classes: {tasks[1]} ...")

    # 3. Initialize Models
    input_dim = train_x.shape[1]
    som_params.input_dim = input_dim

    model = SatSOM(som_params).to(device)
    knn = KNNClassifier(k=5)
    dsdm = DSDMClassifier(input_dim=input_dim, n_classes=n_total_classes, device=device)
    propre = PROPREClassifier(
        input_dim=input_dim, nH=15, n_classes=n_total_classes, device=device
    )

    records = []

    # 4. Continual Learning Loop
    for task_idx, class_group in enumerate(tasks):
        logger.info(f"\n=== Starting Task {task_idx} (Classes: {class_group}) ===")

        # Filter Train Data for current task
        # We only train on images belonging to the current classes
        mask = torch.isin(train_y, torch.tensor(class_group, device=device))
        task_x = train_x[mask]
        task_y = train_y[mask]

        # Shuffle
        perm = torch.randperm(len(task_x))
        task_x = task_x[perm]
        task_y = task_y[perm]

        # Train Loop
        model.train()
        for i, (img, lbl) in enumerate(
            tqdm(zip(task_x, task_y), total=len(task_x), desc="Training")
        ):
            img = img.unsqueeze(0)
            lbl = lbl.unsqueeze(0)

            # SatSOM
            lbl_oh = F.one_hot(lbl, num_classes=n_total_classes).float()
            model.step(img, lbl_oh)

            # Others
            knn.partial_fit(img, lbl)
            dsdm.partial_fit(img, lbl)
            propre.partial_fit(img, lbl)

        # ------------------------------------------------
        # Evaluation
        # ------------------------------------------------
        # In Core50 NC, evaluation is usually done on the FULL test set (all 50 classes)
        # to see how accuracy evolves globally.

        logger.info(f"Evaluating Task {task_idx} on full test set...")

        # Batch evaluation to save GPU memory
        eval_batch_size = 1000
        n_test = len(test_x)

        # Accumulators
        correct_som = 0
        correct_knn = 0
        correct_dsdm = 0
        correct_propre = 0

        with torch.no_grad():
            for i in range(0, n_test, eval_batch_size):
                end = min(i + eval_batch_size, n_test)
                batch_x = test_x[i:end]
                batch_y = test_y[i:end]

                # SatSOM
                preds_som = []
                for bx in batch_x:
                    preds_som.append(model(bx.unsqueeze(0)).argmax().item())
                correct_som += torch.sum(
                    torch.tensor(preds_som, device=device) == batch_y
                ).item()

                # kNN
                p_knn = knn.predict(batch_x)
                correct_knn += (p_knn == batch_y).sum().item()

                # DSDM
                p_dsdm = dsdm.predict(batch_x).argmax(1)
                correct_dsdm += (p_dsdm == batch_y).sum().item()

                # PROPRE
                p_propre = propre.predict(batch_x).argmax(1)
                correct_propre += (p_propre == batch_y).sum().item()

        acc_som = (correct_som / n_test) * 100
        acc_knn = (correct_knn / n_test) * 100
        acc_dsdm = (correct_dsdm / n_test) * 100
        acc_propre = (correct_propre / n_test) * 100

        logger.info(f"Task {task_idx} Result (Acc % on all 50 classes):")
        logger.info(
            f"SOM: {acc_som:.2f} | kNN: {acc_knn:.2f} | DSDM: {acc_dsdm:.2f} | PROPRE: {acc_propre:.2f}"
        )

        records.append(
            {
                "task_id": task_idx,
                "classes_added": str(class_group),
                "som_acc": acc_som,
                "knn_acc": acc_knn,
                "dsdm_acc": acc_dsdm,
                "propre_acc": acc_propre,
            }
        )

        if save_model:
            torch.save(model.state_dict(), out_dir / f"som_task{task_idx}.pth")

    # Save Results
    df = pd.DataFrame(records)
    csv_path = out_dir / "core50_nc_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Benchmark Complete. Results saved to {csv_path}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    params = SatSOMParameters(
        grid_shape=(60, 60),
        input_dim=512,
        output_dim=50,
        initial_lr=0.5,
        initial_sigma=30.0,
        Lr=0.01,
        Lr_bias=0.1,
        Lr_sigma=0.05,
        q=0.005,
        p=10,
    )

    eval_som(
        som_params=params,
        core50_root="./core50_128x128",
        output_path="./output_core50_nc",
    )
