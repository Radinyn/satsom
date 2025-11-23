import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, models
from tqdm.auto import tqdm
from PIL import Image

from satsom.model import SatSOM, SatSOMParameters

# from satsom.visualization import create_satsom_image # Visualizer disabled for embeddings
from satsom.eval.knn import KNNClassifier

from satsom.eval.dsdm import DSDMClassifier
from satsom.eval.propre import PROPREClassifier


# ---------------------------------------------------------
# 1. Feature Extractor Logic
# ---------------------------------------------------------
class Core50ImageDataset(Dataset):
    """
    Helper dataset just to load images for the feature extractor.
    """

    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        if not os.path.exists(root):
            raise FileNotFoundError(f"Core50 root not found: {root}")

        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            p = os.path.join(root, cls)
            if not os.path.isdir(p):
                continue
            for fname in os.listdir(p):
                if fname.lower().endswith(("png", "jpg", "jpeg")):
                    self.samples.append(
                        (os.path.join(p, fname), self.class_to_idx[cls])
                    )

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
    # For ResNet50, output dim is 2048.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Remove the classification head (fc layer)
    # We keep the backbone + average pooling
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    # 3. Create Loader
    raw_dataset = Core50ImageDataset(root, transform=transform)
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


def load_core50_features(
    root: str, device: str, train_split: float = 0.8
) -> Tuple[Dataset, Dataset, int]:
    """
    Loads Core50, converts to embeddings, splits into train/test.
    Returns (TrainSet, TestSet, InputDimension)
    """
    features, labels = get_resnet_embeddings(root, device)

    dataset = TensorDataset(features, labels)

    n = len(dataset)
    n_train = int(n * train_split)
    train_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(0)
    )

    input_dim = features.shape[1]
    return train_set, test_set, input_dim


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------
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

    # Note: We perform extraction on 'device' to speed it up,
    # then move data to CPU RAM, then back to device during training loop
    # to save GPU VRAM for the models.
    train_set, test_set, input_dim = load_core50_features(
        core50_root, device=device, train_split=0.8
    )

    # Override the input dimension in params to match ResNet output (e.g., 512)
    som_params.input_dim = input_dim
    logger.info(f"Input dimension determined from ResNet: {input_dim}")

    # Helper to aggregate data from the subset
    def collect_data(dataset):
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        f_list, l_list = [], []
        for f, lbl in loader:
            f_list.append(f)
            l_list.append(lbl)
        return torch.cat(f_list).to(device), torch.cat(l_list).to(device)

    images, labels = collect_data(train_set)

    # We also need the test set loaded for evaluation
    test_images_all, test_labels_all = collect_data(test_set)

    unique_labels = sorted(labels.unique().tolist())
    n_classes = len(unique_labels)

    size_limit = size_limit or len(images)
    eval_limit = eval_limit or len(images)

    # Organize data by label for the phase creation
    by_label = defaultdict(dict)

    # Split logic:
    # Since we already did a random split in `load_core50_features`,
    # 'train_set' is used for online training, 'test_set' for eval.
    # We just organize 'train_set' by label here for the phasing.

    for lbl in unique_labels:
        # Training Data
        mask_train = labels == lbl
        by_label["train"][lbl] = images[mask_train]

        # Testing Data
        mask_test = test_labels_all == lbl
        by_label["test"][lbl] = test_images_all[mask_test]

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

            # [NOTE] Visualization Disabled
            # Since inputs are ResNet embeddings (vectors), not images,
            # we cannot use create_satsom_image(width=28, height=28).
            # Using it would result in a dimension mismatch or random noise.

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
            # Note: SatSOM needs batch dim
            # We process in batches for speed during eval
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
    # Initial params - input_dim will be overwritten by the loader
    params = SatSOMParameters(
        grid_shape=(60, 60),
        input_dim=512,  # Placeholder, will update automatically
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
        core50_root="./core50",
        output_path="./output_core50",
    )
