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

from satsom.eval.dsdm import DSDMClassifier
from satsom.eval.propre import PROPREClassifier


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
    phases: list[list[int]] = ([0], [1], [2], [4], [5], [6], [7], [8], [3], [9]),
    dataset: EvalDataset = EvalDataset.FashionMNIST,
    dataset_root_dir: Optional[str] = None,
):
    """
    Train SatSOM and compare with: kNN, DSDM, PROPRE.
    """

    assert epochs_per_phase == 1, "`epochs_per_phase != 1` not yet supported"

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # optional images
    if save_images_each_step:
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

    # logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if enable_logging else logging.WARNING)

    # ---------------------------------------------------
    # Data
    # ---------------------------------------------------
    logger.info(f"Loading {dataset.name}...")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    dataset_dir = dataset_root_dir or (out_dir / "data")
    eval_dataset = dataset.value(root=dataset_dir, download=True, transform=transform)

    images = torch.stack([img for img, _ in eval_dataset]).to(device)
    labels = torch.tensor([lbl for _, lbl in eval_dataset], device=device)

    size_limit = size_limit or len(labels)
    eval_limit = eval_limit or len(labels)

    # group by class
    by_label = defaultdict(dict)
    unique_labels = labels.unique(sorted=True).tolist()

    for lbl in unique_labels:
        imgs_lbl = images[labels == lbl]
        n_train = int(len(imgs_lbl) * train_perc)
        by_label["train"][lbl] = imgs_lbl[:n_train]
        by_label["test"][lbl] = imgs_lbl[n_train:]

    # ---------------------------------------------------
    # Models
    # ---------------------------------------------------
    model = SatSOM(som_params).to(device)
    knn = KNNClassifier(k=5)

    input_dim = 28 * 28
    n_classes = len(unique_labels)

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

    # ---------------------------------------------------
    # Training loop
    # ---------------------------------------------------
    records = []

    for phase_idx, phase_labels in enumerate(phases, start=1):
        logger.info(f"--- Phase {phase_idx}: labels {phase_labels} ---")

        for epoch in range(1, epochs_per_phase + 1):
            logger.info(f"Epoch {epoch}/{epochs_per_phase}")

            imgs = torch.cat([by_label["train"][lbl] for lbl in phase_labels], dim=0)
            labs = torch.cat(
                [
                    torch.full((len(by_label["train"][lbl]),), lbl, device=device)
                    for lbl in phase_labels
                ],
                dim=0,
            )

            perm = torch.randperm(len(imgs))[:size_limit]
            imgs = imgs[perm]
            labs = labs[perm]

            labs_oh = torch.nn.functional.one_hot(labs, num_classes=n_classes)

            train_iter = enumerate(zip(imgs, labs_oh), start=1)
            if show_progress:
                train_iter = tqdm(
                    train_iter,
                    total=len(imgs),
                    desc=f"Phase{phase_idx}",
                    leave=False,
                )

            # update knn, som, DSDM, PROPRE
            knn.partial_fit(imgs, labs)

            for i, (img, lbl_oh) in train_iter:
                # --- SatSOM ---
                model.step(img.unsqueeze(0), lbl_oh.unsqueeze(0))

                # --- DSDM ---
                dsdm.partial_fit(img.unsqueeze(0), labs[i - 1 : i])

                # --- PROPRE ---
                propre.partial_fit(img.unsqueeze(0), labs[i - 1 : i])

                # optionally create images of SOM grid
                if save_images_each_step and phase_idx == 1:
                    n_images = 1000
                    if i % max(1, len(imgs) // n_images) == 0:
                        create_satsom_image(
                            model,
                            output_path=out_dir / f"images/phase{phase_idx}_i{i}.png",
                            img_width=28,
                            img_height=28,
                        )

        # Save SOM map for entire phase
        create_satsom_image(
            model,
            output_path=out_dir / f"phase{phase_idx}.jpg",
            img_width=28,
            img_height=28,
        )

        # ---------------------------------------------------
        # Evaluation
        # ---------------------------------------------------
        model.eval()
        logger.info(f"Evaluating after Phase {phase_idx}")

        for lbl in unique_labels:
            test = by_label["test"][lbl][:eval_limit]

            # SOM classification
            acc_som = (
                100
                * sum((model(img.unsqueeze(0)).argmax().item() == lbl) for img in test)
                / len(test)
            )

            # kNN
            knn_pred = knn.predict(test)
            acc_knn = 100 * (knn_pred == lbl).sum().item() / len(test)

            # DSDM
            dsdm_pred = dsdm.predict(test).argmax(1)
            acc_dsdm = 100 * (dsdm_pred == lbl).sum().item() / len(test)

            # PROPRE
            propre_pred = propre.predict(test).argmax(1)
            acc_propre = 100 * (propre_pred == lbl).sum().item() / len(test)

            logger.info(
                f"Label {lbl}: SOM {acc_som:.2f}%, kNN {acc_knn:.2f}%, "
                f"DSDM {acc_dsdm:.2f}%, PROPRE {acc_propre:.2f}%"
            )

            records.append(
                {
                    "phase": phase_idx,
                    "label": lbl,
                    "accuracy_som": acc_som,
                    "accuracy_knn": acc_knn,
                    "accuracy_dsdm": acc_dsdm,
                    "accuracy_propre": acc_propre,
                }
            )

        # Save SOM model for this phase
        if save_model:
            state_dict_path = out_dir / f"som_phase{phase_idx}.pth"
            torch.save(model.state_dict(), state_dict_path)
            logger.info(f"SOM saved to {state_dict_path}")

    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "accuracy.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Accuracy saved at {csv_path}")


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
