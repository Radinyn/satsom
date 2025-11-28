import logging
from pathlib import Path
from collections import defaultdict
from typing import Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# Import from the file we just created
from satsom.model_growing import GrowingSatSOM
from satsom.model import SatSOMParameters


# ---------------------------------------------------------
# 1. Feature Extraction Logic (Reused)
# ---------------------------------------------------------
def extract_cifar_features(
    root: str, device: str, batch_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads CIFAR10 (Train + Test), resizes to 224x224,
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

    # Load Raw Datasets
    # Download=True checks if it exists first
    train_ds = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test_ds = datasets.CIFAR10(
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

    print("Extracting features from CIFAR10...")
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
# 2. Evaluation Logic
# ---------------------------------------------------------
def eval_som(
    som_params: SatSOMParameters,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    train_perc: float = 0.8,
    dataset_root_dir: str = "./data",
):
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup Logging
    logger = logging.getLogger("eval_memory")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    # 1. Load Data
    features, labels = extract_cifar_features(dataset_root_dir, device)

    # Update input dim based on features
    som_params.input_dim = features.shape[1]

    # Move to device
    features = features.to(device)
    labels = labels.to(device)

    # 2. Organize Data Splits
    unique_labels = sorted(labels.unique().tolist())
    n_classes = len(unique_labels)

    by_label = defaultdict(dict)
    for lbl in unique_labels:
        mask = labels == lbl
        imgs_lbl = features[mask]
        n_train = int(train_perc * len(imgs_lbl))
        by_label["train"][lbl] = imgs_lbl[:n_train]
        by_label["test"][lbl] = imgs_lbl[n_train:]

    # Define Phases (1 class per phase)
    phases = [[lbl] for lbl in unique_labels]

    # 3. Define Configurations
    configs = []
    centering_options = [True, False]

    # Naive Growing
    for centering in centering_options:
        c_name = "Centered" if centering else "Uncentered"
        configs.append(
            {
                "name": f"Naive_{c_name}",
                "naive": True,
                "centering": centering,
                "scale": 0.0,  # Irrelevant for naive
            }
        )

    # Smart Growing with scales [0.0, ..., 1.0]
    scales = [round(x * 0.1, 1) for x in range(11)]  # 0.0 to 1.0
    for s in scales:
        for centering in centering_options:
            c_name = "Centered" if centering else "Uncentered"
            configs.append(
                {
                    "name": f"Smart_Scale_{s}_{c_name}",
                    "naive": False,
                    "centering": centering,
                    "scale": s,
                }
            )

    logger.info(f"Total configurations to run: {len(configs)}")

    all_records = []

    # 4. Main Experiment Loop
    for conf in configs:
        logger.info(f"--- Running Configuration: {conf['name']} ---")

        # Reset params for each run to ensure clean state
        run_params = copy.deepcopy(som_params)

        # Initialize Model
        model = GrowingSatSOM(
            params=run_params,
            naive=conf["naive"],
            centering=conf["centering"],  # Pass centering configuration
            radius_threshold_scale=conf["scale"],
        ).to(device)

        # Phase Loop
        seen_labels = []

        for phase_idx, phase_labels in enumerate(phases, start=1):
            seen_labels.extend(phase_labels)

            # Prepare Training Data for Phase
            imgs_list = [by_label["train"][lbl] for lbl in phase_labels]
            if not imgs_list:
                continue

            train_imgs = torch.cat(imgs_list)
            train_lbls = torch.cat(
                [
                    torch.full((len(by_label["train"][lbl]),), lbl, device=device)
                    for lbl in phase_labels
                ]
            )

            # Shuffle
            perm = torch.randperm(len(train_imgs))
            train_imgs = train_imgs[perm]
            train_lbls = train_lbls[perm]

            # Train
            model.train()

            pbar = tqdm(
                zip(train_imgs, train_lbls),
                total=len(train_imgs),
                desc=f"{conf['name']} | Phase {phase_idx}",
                leave=False,
            )

            for img, lbl in pbar:
                lbl_oh = F.one_hot(lbl, num_classes=n_classes).float()
                # model.step handles both grow_map and SatSOM update
                model.step(img.unsqueeze(0), lbl_oh.unsqueeze(0))

            # --- End of Phase Measurement ---

            # 1. Network Size
            current_grid = model.satsom.params.grid_shape
            network_size = int(np.prod(current_grid))

            # 2. Evaluation (Accuracy on all seen classes so far)

            for lbl in unique_labels:
                # Only evaluate on labels we have test data for
                if lbl not in by_label["test"] or len(by_label["test"][lbl]) == 0:
                    continue

                test_imgs = by_label["test"][lbl]

                # Predict
                model.eval()
                with torch.no_grad():
                    preds = model(test_imgs)  # (N, n_classes)
                pred_lbls = preds.argmax(dim=1)

                acc = (pred_lbls == lbl).float().mean().item() * 100.0

                record = {
                    "config": conf["name"],
                    "naive_growing": conf["naive"],
                    "centering_enabled": conf["centering"],  # Added to CSV
                    "radius_threshold_scale": conf["scale"],
                    "phase": phase_idx,
                    "phase_labels": str(phase_labels),
                    "network_neurons": network_size,
                    "grid_h": current_grid[0],
                    "grid_w": current_grid[1],
                    "eval_label": lbl,
                    "accuracy": acc,
                }
                all_records.append(record)

            logger.info(
                f"Phase {phase_idx} Complete. Size: {current_grid} ({network_size} neurons). Average accuracy: {acc:.2f}%"
            )

    # 5. Save Results
    df = pd.DataFrame(all_records)
    csv_path = out_dir / "accuracy_cifar10_memory.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark results to {csv_path}")
    print(df.head())


if __name__ == "__main__":
    # Parameters
    # We use a smaller initial grid shape (e.g., 10x10) so growth is visible and necessary.
    params = SatSOMParameters(
        grid_shape=(10, 10),
        input_dim=512,  # Will be overwritten by feature extractor
        output_dim=10,
        initial_lr=0.5,
        initial_sigma=3.0,  # Smaller sigma for smaller initial map
        Lr=0.01,
        Lr_bias=0.2,
        Lr_sigma=0.05,
        q=0.005,
        p=10.0,
    )

    eval_som(som_params=params, output_path="./output_cifar10_growing", train_perc=0.8)
