import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm

from satsom.model import SatSOM, SatSOMParameters
from satsom.eval.knn import KNNClassifier
from satsom.eval.dsdm import DSDMClassifier


# ---------------------------------------------------------
# 1. Feature Extraction Logic
# ---------------------------------------------------------
def extract_cifar_features(
    root: str, device: str, batch_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads CIFAR10, splits into Train (50k) and Test (10k),
    resizes to 224x224, passes through ResNet18, returns embeddings.

    Returns:
        train_features, train_labels, test_features, test_labels
    """
    print(f"Preparing ResNet18 feature extractor on {device}...")

    # Standard ImageNet preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load Raw Datasets separately to maintain strict Train/Test split
    train_ds = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test_ds = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )

    # Prepare Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()  # Remove classification head
    model.to(device)
    model.eval()

    def extract_from_loader(loader):
        feats_list = []
        labels_list = []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Extracting"):
                imgs = imgs.to(device)
                feats = model(imgs)
                feats = feats.view(feats.size(0), -1)
                feats_list.append(feats.cpu())
                labels_list.append(lbls)
        return torch.cat(feats_list), torch.cat(labels_list)

    print("Extracting Training Features...")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )
    train_X, train_Y = extract_from_loader(train_loader)

    print("Extracting Test Features...")
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_X, test_Y = extract_from_loader(test_loader)

    print("Extraction complete.")
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")

    return train_X, train_Y, test_X, test_Y


# ---------------------------------------------------------
# 2. Evaluation Loop (Decreasing Train %)
# ---------------------------------------------------------
def eval_som(
    som_params: SatSOMParameters,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dataset_root_dir: Optional[str] = None,
):
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging setup
    logger = logging.getLogger("data_efficiency")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    # 1. Load Data
    dataset_dir = dataset_root_dir or (out_dir / "data")
    train_X_full, train_Y_full, test_X, test_Y = extract_cifar_features(
        str(dataset_dir), device
    )

    # Auto-detect input dim
    input_dim = train_X_full.shape[1]
    som_params.input_dim = input_dim
    n_classes = len(train_Y_full.unique())

    # Move Test Set to Device (for faster repeated evaluation)
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    # Shuffle training indices once so "80%" and "70%" are drawn from the same distribution
    # (though typically we just take the first N after shuffling)
    total_train_samples = len(train_X_full)
    perm = torch.randperm(total_train_samples)
    train_X_shuffled = train_X_full[perm]
    train_Y_shuffled = train_Y_full[perm]

    # Percentages: 80% down to 10%
    percentages = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    results = []

    for pct in percentages:
        n_samples = int(total_train_samples * pct)
        logger.info(
            f"=== Starting Run: Train Percentage {pct * 100:.0f}% ({n_samples} samples) ==="
        )

        # Subset data
        current_train_X = train_X_shuffled[:n_samples].to(device)
        current_train_Y = train_Y_shuffled[:n_samples].to(device)

        # -------------------------------------------------
        # Initialize Fresh Models for this percentage
        # -------------------------------------------------
        # SatSOM
        model = SatSOM(som_params).to(device)

        # kNN
        knn = KNNClassifier(k=5)

        # DSDM
        dsdm = DSDMClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            T=2.0,
            ema=0.02,
            pruning=False,
            device=device,
        )

        # -------------------------------------------------
        # Incremental Training Loop
        # -------------------------------------------------
        # Train strictly incrementally (sample by sample or batch by batch)
        # Using a simple loop here.

        model.train()

        for i in tqdm(range(n_samples), desc=f"Training {pct * 100:.0f}%"):
            img = current_train_X[i].unsqueeze(0)  # (1, 512)
            lbl = current_train_Y[i].unsqueeze(0)  # (1)

            lbl_oh = F.one_hot(lbl, num_classes=n_classes).float()

            # SatSOM Step
            model.step(img, lbl_oh)

            # kNN Step
            knn.partial_fit(img, lbl)

            # DSDM Step
            dsdm.partial_fit(img, lbl)

        # -------------------------------------------------
        # Evaluation on Constant Test Set
        # -------------------------------------------------
        logger.info(f"Evaluating models on full Test Set ({len(test_X)} samples)...")

        # SOM (batch evaluation for speed)
        preds_som = []
        # Evaluate in small chunks if memory is tight, though 512 floats is small
        # Simple loop:
        for t_img in test_X:
            preds_som.append(model(t_img.unsqueeze(0)).argmax().item())
        acc_som = (torch.Tensor(preds_som) == test_Y).float().mean().item() * 100

        # kNN Eval
        pred_knn = knn.predict(test_X)
        acc_knn = (pred_knn == test_Y).float().mean().item() * 100

        # DSDM Eval
        # DSDM usually requires batching if memory is tight, but 10k x 512 fits on GPU
        pred_dsdm_logits = dsdm.predict(test_X)
        pred_dsdm = pred_dsdm_logits.argmax(dim=1)
        acc_dsdm = (pred_dsdm == test_Y).float().mean().item() * 100

        logger.info(
            f"Result {pct * 100:.0f}% Data: SOM={acc_som:.2f}% | kNN={acc_knn:.2f}% | DSDM={acc_dsdm:.2f}%"
        )

        results.append(
            {
                "train_percentage": pct,
                "n_samples": n_samples,
                "acc_som": acc_som,
                "acc_knn": acc_knn,
                "acc_dsdm": acc_dsdm,
            }
        )

    # -------------------------------------------------
    # Save Results
    # -------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = out_dir / "cifar10_data_efficiency_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Experiment complete. Results saved to {csv_path}")
    print(df)


if __name__ == "__main__":
    # Parameters setup
    params = SatSOMParameters(
        grid_shape=(60, 60),
        input_dim=512,  # Will be auto-updated
        output_dim=10,
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
        output_path="./output_cifar10_efficiency",
    )
