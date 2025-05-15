import logging
from pathlib import Path
from collections import defaultdict
from enum import Enum
from typing import Optional

import torch
import pandas as pd
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from satsom.model import SatSOM, SatSOMParameters
from satsom.visualization import create_satsom_image
from satsom.eval.knn import KNNClassifier
from satsom.eval.ewc import MLP_EWC, CNN_EWC


class EvalDataset(Enum):
    MNIST = datasets.MNIST
    FashionMNIST = datasets.FashionMNIST
    KMNIST = datasets.KMNIST


def eval_som(
    som_params: SatSOMParameters,
    output_path: str,
    device: str = "mps",
    save_images_each_step: bool = False,
    save_model: bool = False,
    enable_logging: bool = True,
    show_progress: bool = True,
    train_perc: float = 0.8,
    epochs_per_phase: int = 1,
    size_limit: Optional[int] = None,
    eval_limit: Optional[int] = None,
    phases: list[list[int]] = ([0, 1, 2, 4, 5, 6, 7, 8], [3, 9]),
    dataset: EvalDataset = EvalDataset.FashionMNIST,
    dataset_root_dir: Optional[str] = None,
):
    """
    Train a SatSOM and compare it to kNN, optionally logging & saving SOM maps.

    Args:
        som_params: pre-configured SatSOMParameters.
        output_path: directory where checkpoints, images, and state-dict go.
        device: torch device identifier (e.g. "cpu", "cuda", "mps").
        save_images_each_step: if True, will call `create_satsom_image(...)` at designated steps.
        enable_logging: if False, suppresses INFO-level logs.
        show_progress: if False, disables tqdm bars.
        train_perc: fraction of each class to use for training.
        epochs_per_phase: number of epochs per phase.
        size_limit: max number of samples per epoch.
        eval_limit: max test images per class for evaluation.
        phases: list of label-groups for phased training.
    """
    assert epochs_per_phase == 1, "`epochs_per_phase != 1` is not yet supported"

    # ─── Setup ───────────────────────────────────────────────────────────────────
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_images_each_step:
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if enable_logging else logging.WARNING)

    # ─── Data Loading & Split ───────────────────────────────────────────────────
    logger.info(f"Loading {dataset.name}...")
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # for RGB datasets
            transforms.Resize((28, 28)),  # for non-MNIST datasets
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    dataset_dir = dataset_root_dir or (out_dir / "data")
    eval_dataset = dataset.value(root=dataset_dir, download=True, transform=transform)
    images = torch.stack([img for img, _ in eval_dataset]).to(device)
    labels = torch.tensor([lbl for _, lbl in eval_dataset], device=device)

    # If there is no limit use the whole dataset
    size_limit = size_limit or len(labels)
    eval_limit = eval_limit or len(labels)

    by_label = defaultdict(dict)
    unique_labels = labels.unique(sorted=True).tolist()
    for lbl in unique_labels:
        imgs_lbl = images[labels == lbl]
        n_train = int(len(imgs_lbl) * train_perc)
        by_label["train"][lbl] = imgs_lbl[:n_train]
        by_label["test"][lbl] = imgs_lbl[n_train:]

    # ─── Models Setup ──────────────────────────────────────────────────────
    model = SatSOM(som_params).to(device)
    knn = KNNClassifier(k=5)

    # EWC models that run for 1 epoch
    mlp_ewc = MLP_EWC(device=device)
    cnn_ewc = CNN_EWC(device=device)

    # EWC models that run for 10 epochs
    mlp_ewc10 = MLP_EWC(device=device)
    cnn_ewc10 = CNN_EWC(device=device)

    opt_mlp = torch.optim.Adam(mlp_ewc.parameters(), lr=1e-4)
    opt_cnn = torch.optim.Adam(cnn_ewc.parameters(), lr=1e-4)
    opt_mlp10 = torch.optim.Adam(mlp_ewc10.parameters(), lr=1e-4)
    opt_cnn10 = torch.optim.Adam(cnn_ewc10.parameters(), lr=1e-4)

    # ─── Training / Evaluation Loop ─────────────────────────────────────────────
    records = []
    for phase_idx, phase_labels in enumerate(phases, start=1):
        logger.info(f"--- Phase {phase_idx}: training on labels {phase_labels} ---")

        for epoch in range(1, epochs_per_phase + 1):
            logger.info(f"Epoch {epoch}/{epochs_per_phase}")
            # assemble epoch data
            imgs = torch.cat([by_label["train"][lbl] for lbl in phase_labels], dim=0)
            labs = torch.cat(
                [
                    torch.full((len(by_label["train"][lbl]),), lbl, device=device)
                    for lbl in phase_labels
                ],
                dim=0,
            )
            perm = torch.randperm(len(imgs))[:size_limit]
            imgs, labs = imgs[perm], labs[perm].unsqueeze(1)
            labs_oh = torch.nn.functional.one_hot(labs.squeeze(), num_classes=10)

            knn.partial_fit(imgs, labs.squeeze())

            model.train()
            train_iter = enumerate(zip(imgs, labs_oh), start=1)
            if show_progress:
                train_iter = tqdm(
                    train_iter,
                    total=len(imgs),
                    desc=f"Phase{phase_idx}-E{epoch}",
                    leave=False,
                )
            for i, (img, lbl_oh) in train_iter:
                model.step(img.unsqueeze(0), lbl_oh.unsqueeze(0))

                # Create 1000 images
                n_images = 1000
                if (
                    save_images_each_step
                    and phase_idx == 1
                    and i % max(1, len(imgs) // n_images) == 0
                ):
                    create_satsom_image(
                        model,
                        output_path=out_dir
                        / f"images/phase{phase_idx}_epoch{epoch}_{i}.png",
                        img_width=28,
                        img_height=28,
                    )

        # Train EWC models on this phase data
        batch_data = imgs.view(-1, 1, 28, 28)
        batch_flat = batch_data.view(batch_data.size(0), -1)
        for model_obj, opt_obj, ewc_epochs, name in [
            (mlp_ewc, opt_mlp, 1, "mlp_ewc"),
            (cnn_ewc, opt_cnn, 1, "cnn_ewc"),
            (mlp_ewc10, opt_mlp10, 10, "mlp_ewc10"),
            (cnn_ewc10, opt_cnn10, 10, "cnn_ewc10"),
        ]:
            if "mlp" in name:
                model_obj.partial_fit(
                    opt_obj, batch_flat, labs.squeeze(1), epochs=ewc_epochs
                )
            else:
                model_obj.partial_fit(
                    opt_obj, batch_data, labs.squeeze(1), epochs=ewc_epochs
                )

        # ─── Evaluation ─────────────────────────────────────────────────────────
        create_satsom_image(
            model,
            output_path=out_dir / f"phase{phase_idx}.jpg",
            img_width=28,
            img_height=28,
        )

        logger.info(f"Evaluating after Phase {phase_idx}")
        model.eval()
        for lbl in unique_labels:
            test = by_label["test"][lbl][:eval_limit]
            # SOM
            correct_som = sum(
                (model(img.unsqueeze(0)).argmax().item() == lbl) for img in test
            )
            acc_som = 100 * correct_som / len(test)
            # kNN
            knn_preds = knn.predict(test)
            acc_knn = 100 * (knn_preds == lbl).sum().item() / len(test)
            # EWC models
            # prepare test tensors
            test_flat = test
            test_cnn = test.view(-1, 1, 28, 28)
            acc_mlp = (
                100 * (mlp_ewc(test_flat).argmax(dim=1) == lbl).sum().item() / len(test)
            )
            acc_cnn = (
                100 * (cnn_ewc(test_cnn).argmax(dim=1) == lbl).sum().item() / len(test)
            )
            acc_mlp10 = (
                100
                * (mlp_ewc10(test_flat).argmax(dim=1) == lbl).sum().item()
                / len(test)
            )
            acc_cnn10 = (
                100
                * (cnn_ewc10(test_cnn).argmax(dim=1) == lbl).sum().item()
                / len(test)
            )

            logger.info(
                f"Label {lbl}: SOM {acc_som:.2f}%, kNN {acc_knn:.2f}%, "
                f"MLP_EWC {acc_mlp:.2f}%, CNN_EWC {acc_cnn:.2f}%, "
                f"MLP_EWC10 {acc_mlp10:.2f}%, CNN_EWC10 {acc_cnn10:.2f}%"
            )

            records.append(
                {
                    "phase": phase_idx,
                    "label": lbl,
                    "accuracy_som": acc_som,
                    "accuracy_knn": acc_knn,
                    "accuracy_mlp_ewc": acc_mlp,
                    "accuracy_cnn_ewc": acc_cnn,
                    "accuracy_mlp_ewc10": acc_mlp10,
                    "accuracy_cnn_ewc10": acc_cnn10,
                }
            )

        # ─── Phase Save ─────────────────────────────────────────────────────────────
        state_dict_path = out_dir / f"som_model_phase{phase_idx}.pth"
        if save_model:
            torch.save(model.state_dict(), state_dict_path)
            logger.info(f"SOM state_dict saved to {state_dict_path}")

    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "accuracy.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Accuracy CSV saved to {csv_path}")


if __name__ == "__main__":
    params = SatSOMParameters(
        grid_shape=(100, 100),
        input_dim=28 * 28,
        output_dim=10,
        initial_lr=0.5,
        initial_sigma=50.0,
        Lr=0.01,
        Lr_bias=0.2,
        Lr_sigma=0.05,
        q=0.005,
        p=10.0,
    )

    eval_som(
        som_params=params,
        output_path="./output",
        size_limit=10_000,
        eval_limit=1_000,
        dataset=EvalDataset.FashionMNIST,
    )
