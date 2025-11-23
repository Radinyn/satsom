import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm.auto import tqdm
from PIL import Image

# Assuming these exist in your environment
from satsom.model import SatSOM, SatSOMParameters
from satsom.eval.knn import KNNClassifier

from satsom.eval.dsdm import DSDMClassifier
from satsom.eval.propre import PROPREClassifier


# ---------------------------------------------------------
# 1. Feature Extractor Logic (Updated for sX/oY structure)
# ---------------------------------------------------------
class Core50ImageDataset(Dataset):
    """
    Custom loader for Core50 structure:
       root/
         s1/
           o1/ -> images
           o2/ -> images
         s2/
           o1/ ...

    It aggregates all 'o1' folders from all sessions into Class 0,
    all 'o2' folders into Class 1, etc.
    """

    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        if not os.path.exists(root):
            raise FileNotFoundError(f"Core50 root not found: {root}")

        # 1. Scan for all unique Object names (o1, o2...) to build class index
        object_names = set()

        # Traverse sessions (s1, s2, ...)
        for session_name in os.listdir(root):
            session_path = os.path.join(root, session_name)
            if not os.path.isdir(session_path):
                continue

            # Traverse objects inside session
            for obj_name in os.listdir(session_path):
                if obj_name.startswith("o") and os.path.isdir(
                    os.path.join(session_path, obj_name)
                ):
                    object_names.add(obj_name)

        # Sort objects naturally (o1, o2, ... o10, not o1, o10, o11, o2)
        # We assume folders are named 'o{number}'
        try:
            sorted_objects = sorted(
                list(object_names), key=lambda x: int(x.replace("o", ""))
            )
        except ValueError:
            # Fallback if folders aren't strictly o{int}
            sorted_objects = sorted(list(object_names))

        self.class_to_idx = {o: i for i, o in enumerate(sorted_objects)}
        print(f"Found {len(self.class_to_idx)} unique classes across sessions.")

        # 2. Collect all images
        for session_name in os.listdir(root):
            session_path = os.path.join(root, session_name)
            if not os.path.isdir(session_path):
                continue

            for obj_name in os.listdir(session_path):
                obj_path = os.path.join(session_path, obj_name)
                if not os.path.isdir(obj_path):
                    continue

                # Get label ID
                if obj_name not in self.class_to_idx:
                    continue
                lbl = self.class_to_idx[obj_name]

                # Collect images
                for fname in os.listdir(obj_path):
                    if fname.lower().endswith(("png", "jpg", "jpeg")):
                        self.samples.append((os.path.join(obj_path, fname), lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lbl


def get_resnet_embeddings(
    root: str, device: str, batch_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterates through the Core50 folder, passes images through ResNet18,
    and returns (Embeddings, Labels).
    """
    print(f"Generating embeddings using ResNet18 on {device}...")

    # 1. Setup Preprocessing for ResNet (ImageNet Stats)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ResNet standard input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 2. Load Model
    # Using ResNet18. Output dim will be 512.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    # 3. Create Loader
    # This now uses the nested sX/oY logic
    raw_dataset = Core50ImageDataset(root, transform=transform)

    if len(raw_dataset) == 0:
        raise ValueError(
            f"No images found in {root}. Check structure (root/sX/oY/img.png)"
        )

    loader = DataLoader(
        raw_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    all_feats = []
    all_labels = []

    # 4. Inference Loop
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting Features"):
            imgs = imgs.to(device)
            # Forward pass
            feats = model(imgs)
            # Flatten just in case (B, 512)
            feats = feats.view(feats.size(0), -1)

            all_feats.append(feats.cpu())
            all_labels.append(lbls)

    features = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    print(f"Feature extraction complete. Shape: {features.shape}")
    return features, labels


def eval_som(
    som_params: SatSOMParameters,
    core50_root: str,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    train_perc: float = 0.8,
    size_limit: Optional[int] = None,
    eval_limit: Optional[int] = None,
    phases: Optional[list[list[int]]] = None,
    save_model: bool = False,
    show_progress: bool = True,
):
    """
    CORE50 continual learning evaluation on EMBEDDINGS.
    """

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("eval_core50")
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # ----------------------------------------------
    # Load data (Features)
    # ----------------------------------------------
    logger.info("Loading CORE50 and extracting ResNet features...")

    features, labels = get_resnet_embeddings(core50_root, device)

    # Auto-detect dimension (512 for ResNet18)
    input_dim = features.shape[1]
    som_params.input_dim = input_dim
    logger.info(f"Input dimension determined from ResNet: {input_dim}")

    # Move data to GPU for training (Core50 is ~128k images, embeddings fit in VRAM)
    # If OOM occurs, keep on CPU and move batches manually
    images = features.to(device)
    labels = labels.to(device)

    unique_labels = sorted(labels.unique().tolist())
    n_classes = len(unique_labels)

    size_limit = size_limit or len(images)
    eval_limit = eval_limit or len(images)

    # Organize data by label for the phase creation
    by_label = defaultdict(dict)

    # Split logic:
    # Core50 is usually trained by Session (Dom-IL) or Class (Class-IL).
    # This script does Class-IL (splitting each class into train/test randomly).
    # If you want strict Session-based splitting (train on s1, test on s2),
    # logic needs changing, but standard Class-IL is random split per class.

    for lbl in unique_labels:
        mask = labels == lbl
        imgs_lbl = images[mask]

        # Shuffle for random split
        perm = torch.randperm(len(imgs_lbl))
        imgs_lbl = imgs_lbl[perm]

        n_train = int(len(imgs_lbl) * train_perc)
        by_label["train"][lbl] = imgs_lbl[:n_train]
        by_label["test"][lbl] = imgs_lbl[n_train:]

    # default phasing: each class one by one
    if phases is None:
        phases = [[lbl] for lbl in unique_labels]

    # ----------------------------------------------
    # Models
    # ----------------------------------------------
    logger.info("Initializing models...")
    model = SatSOM(som_params).to(device)
    knn = KNNClassifier(k=5)

    dsdm = DSDMClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        T=2.0,
        ema=0.02,
        pruning=False,
        device=device,
    )

    propre = PROPREClassifier(
        input_dim=input_dim,
        nH=15,
        n_classes=n_classes,
        device=device,
        lr_som=0.05,
        lr_lr=0.1,
        kappa=1.0,
        theta=0.6,
        p=10,
    )

    # ----------------------------------------------
    # Training loop
    # ----------------------------------------------
    records = []

    for phase_idx, phase_labels in enumerate(phases, start=1):
        logger.info(f"=== Phase {phase_idx}: labels {phase_labels} ===")

        # Gather training data for this phase
        imgs_list = [
            by_label["train"][lbl] for lbl in phase_labels if lbl in by_label["train"]
        ]

        if not imgs_list:
            logger.warning(f"No data found for labels {phase_labels}, skipping phase.")
            continue

        imgs = torch.cat(imgs_list)
        labs = torch.cat(
            [
                torch.full((len(by_label["train"][lbl]),), lbl, device=device)
                for lbl in phase_labels
                if lbl in by_label["train"]
            ]
        )

        # Shuffle training data
        perm = torch.randperm(len(imgs))[:size_limit]
        imgs = imgs[perm]
        labs = labs[perm]

        n_samples = len(imgs)
        iterator = enumerate(zip(imgs, labs), start=1)

        if show_progress:
            iterator = tqdm(iterator, total=n_samples, desc=f"Phase {phase_idx}")

        # update models
        for i, (img, label) in iterator:
            # SatSOM
            lbl_oh = F.one_hot(label, num_classes=n_classes).float()
            model.step(img.unsqueeze(0), lbl_oh.unsqueeze(0))

            # kNN
            knn.partial_fit(img.unsqueeze(0), label.unsqueeze(0))

            # DSDM
            dsdm.partial_fit(img.unsqueeze(0), label.unsqueeze(0))

            # PROPRE
            propre.partial_fit(img.unsqueeze(0), label.unsqueeze(0))

        # ----------------------------------------------
        # Evaluation
        # ----------------------------------------------
        logger.info(f"Evaluating Phase {phase_idx}...")

        for lbl in unique_labels:
            if lbl not in by_label["test"]:
                continue

            test_imgs = by_label["test"][lbl][:eval_limit]
            if len(test_imgs) == 0:
                continue

            # SOM
            preds_som = []
            for t_img in test_imgs:
                preds_som.append(model(t_img.unsqueeze(0)).argmax().item())

            acc_som = 100 * sum(p == lbl for p in preds_som) / len(preds_som)

            # kNN
            knn_pred = knn.predict(test_imgs)
            acc_knn = 100 * (knn_pred == lbl).sum().item() / len(test_imgs)

            # DSDM
            dsdm_pred = dsdm.predict(test_imgs).argmax(1)
            acc_dsdm = 100 * (dsdm_pred == lbl).sum().item() / len(test_imgs)

            # PROPRE
            propre_pred = propre.predict(test_imgs).argmax(1)
            acc_propre = 100 * (propre_pred == lbl).sum().item() / len(test_imgs)

            logger.info(
                f"Class {lbl}: SOM {acc_som:.2f}% | kNN {acc_knn:.2f}% "
                f"| DSDM {acc_dsdm:.2f}% | PROPRE {acc_propre:.2f}%"
            )

            records.append(
                {
                    "phase": phase_idx,
                    "label": lbl,
                    "som": acc_som,
                    "knn": acc_knn,
                    "dsdm": acc_dsdm,
                    "propre": acc_propre,
                }
            )

        if save_model:
            torch.save(model.state_dict(), out_dir / f"som_phase{phase_idx}.pth")

    # save CSV
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "accuracy_core50_resnet.csv", index=False)
    logger.info("Saved accuracy_core50_resnet.csv")


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
        output_path="./output_core50",
        # size_limit=20000,
        # eval_limit=2000,
    )
