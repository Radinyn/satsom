import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm

from satsom.model import SatSOM, SatSOMParameters

# from satsom.visualization import create_satsom_image # Disabled for embeddings
from satsom.eval.knn import KNNClassifier

from satsom.eval.dsdm import DSDMClassifier
from satsom.eval.propre import PROPREClassifier


# ---------------------------------------------------------
# 1. Feature Extraction Logic
# ---------------------------------------------------------
def extract_cifar100_features(
    root: str, device: str, batch_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads CIFAR100 (Train + Test), resizes to 224x224,
    passes through ResNet18 (ImageNet weights), returns embeddings.
    """
    print(f"Preparing ResNet18 feature extractor on {device}...")

    # Standard ImageNet preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Up-sample CIFAR(32) to ResNet(224)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load Raw Datasets (Download if needed)
    train_ds = datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform
    )
    test_ds = datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform
    )

    # Combined loader for extraction
    full_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
    loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Prepare Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()  # Remove classification head
    model.to(device)
    model.eval()

    all_feats = []
    all_labels = []

    print("Extracting features from CIFAR100...")
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="ResNet Inference"):
            imgs = imgs.to(device)
            feats = model(imgs)
            # Flatten (B, 512)
            feats = feats.view(feats.size(0), -1)

            all_feats.append(feats.cpu())
            all_labels.append(lbls)

    features = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    print(f"Extraction complete. Features shape: {features.shape}")
    return features, labels


# ---------------------------------------------------------
# 2. Evaluation Loop
# ---------------------------------------------------------
def eval_som(
    som_params: SatSOMParameters,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_model: bool = False,
    enable_logging: bool = True,
    show_progress: bool = True,
    train_perc: float = 0.8,
    epochs_per_phase: int = 1,
    size_limit: Optional[int] = None,
    eval_limit: Optional[int] = None,
    phases: Optional[list[list[int]]] = None,
    dataset_root_dir: Optional[str] = None,
):
    """
    Train SatSOM, DSDM, PROPRE on CIFAR100 Embeddings with incremental phases.
    """

    assert epochs_per_phase == 1, "`epochs_per_phase != 1` is not supported"

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging setup
    logger = logging.getLogger("eval_cifar100")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if enable_logging else logging.WARNING)

    # ----------------------------------------------
    # Load CIFAR100 Features
    # ----------------------------------------------
    logger.info("Loading CIFAR100 Features...")

    dataset_dir = dataset_root_dir or (out_dir / "data")

    # Extract features using ResNet
    features, labels = extract_cifar100_features(str(dataset_dir), device=device)

    # Move to device (CIFAR100 embeddings fit easily in VRAM)
    images = features.to(device)
    labels = labels.to(device)

    # Auto-detect input dim
    input_dim = images.shape[1]
    som_params.input_dim = input_dim  # Update params dynamically

    size_limit = size_limit or len(images)
    eval_limit = eval_limit or len(images)

    # split by class
    unique_labels = sorted(labels.unique().tolist())
    n_classes = len(unique_labels)

    by_label = defaultdict(dict)
    for lbl in unique_labels:
        mask = labels == lbl
        imgs_lbl = images[mask]

        n_train = int(train_perc * len(imgs_lbl))

        by_label["train"][lbl] = imgs_lbl[:n_train]
        by_label["test"][lbl] = imgs_lbl[n_train:]

    # default: each class is a phase
    if phases is None:
        phases = [[lbl] for lbl in unique_labels]

    # ----------------------------------------------
    # Models
    # ----------------------------------------------
    logger.info(f"Initializing models with input_dim={input_dim}...")

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
        nH=20,
        n_classes=n_classes,
        device=device,
        lr_som=0.05,
        lr_lr=0.1,
        kappa=1.0,
        theta=0.6,
        p=10,
    )

    # ----------------------------------------------
    # Training loop by phase
    # ----------------------------------------------
    records = []

    for phase_idx, phase_labels in enumerate(phases, start=1):
        logger.info(f"=== Phase {phase_idx} | labels {phase_labels} ===")

        # Gather data
        imgs_list = [
            by_label["train"][lbl] for lbl in phase_labels if lbl in by_label["train"]
        ]

        if not imgs_list:
            logger.warning("No data for this phase.")
            continue

        imgs = torch.cat(imgs_list)
        labs = torch.cat(
            [
                torch.full((len(by_label["train"][lbl]),), lbl, device=device)
                for lbl in phase_labels
                if lbl in by_label["train"]
            ]
        )

        perm = torch.randperm(len(imgs))[:size_limit]
        imgs = imgs[perm]
        labs = labs[perm]

        train_iter = enumerate(zip(imgs, labs), start=1)
        if show_progress:
            train_iter = tqdm(train_iter, total=len(imgs), desc=f"Phase {phase_idx}")

        for i, (img, lbl) in train_iter:
            lbl_oh = F.one_hot(lbl, num_classes=n_classes).float()

            # SatSOM update
            model.step(img.unsqueeze(0), lbl_oh.unsqueeze(0))

            # kNN
            knn.partial_fit(img.unsqueeze(0), lbl.unsqueeze(0))

            # DSDM
            dsdm.partial_fit(img.unsqueeze(0), lbl.unsqueeze(0))

            # PROPRE
            propre.partial_fit(img.unsqueeze(0), lbl.unsqueeze(0))

            # [NOTE] Visualizations disabled
            # Inputs are 512-dim vectors, not images.

        # ----------------------------------------------
        # Evaluation after the phase
        # ----------------------------------------------
        logger.info(f"Evaluating after Phase {phase_idx}...")

        for lbl in unique_labels:
            if lbl not in by_label["test"]:
                continue

            test_imgs = by_label["test"][lbl][:eval_limit]
            if len(test_imgs) == 0:
                continue

            # SOM
            # Process in a loop or batches if necessary, but 512-dim is small enough
            preds_som = []
            for t_img in test_imgs:
                preds_som.append(model(t_img.unsqueeze(0)).argmax().item())

            acc_som = 100 * sum(p == lbl for p in preds_som) / len(test_imgs)

            # kNN
            pred_knn = knn.predict(test_imgs)
            acc_knn = 100 * (pred_knn == lbl).sum().item() / len(test_imgs)

            # DSDM
            pred_dsdm = dsdm.predict(test_imgs).argmax(1)
            acc_dsdm = 100 * (pred_dsdm == lbl).sum().item() / len(test_imgs)

            # PROPRE
            pred_propre = propre.predict(test_imgs).argmax(1)
            acc_propre = 100 * (pred_propre == lbl).sum().item() / len(test_imgs)

            logger.info(
                f"Label {lbl}:  SOM {acc_som:.2f}% | kNN {acc_knn:.2f}% "
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

    # ----------------------------------------------
    # Save CSV
    # ----------------------------------------------
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "accuracy_cifar100_resnet.csv", index=False)
    logger.info("Saved accuracy_cifar100_resnet.csv")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    params = SatSOMParameters(
        grid_shape=(60, 60),
        input_dim=512,  # Placeholder, will be autoset to 512
        output_dim=100,  # CIFAR100 has 100 classes
        initial_lr=0.5,
        initial_sigma=30.0,
        Lr=0.01,
        Lr_bias=0.2,
        Lr_sigma=0.05,
        q=0.005,
        p=10.0,
    )

    eval_som(
        som_params=params,
        output_path="./output_cifar100",
    )
