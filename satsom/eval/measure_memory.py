import logging
from pathlib import Path
from collections import defaultdict
from enum import Enum
from typing import Optional
import torch
import pandas as pd
from torchvision import datasets, transforms

from satsom.model import SatSOM, SatSOMParameters
from satsom.visualization import create_satsom_image
from satsom.eval.knn import KNNClassifier
from satsom.eval.dsdm import DSDMClassifier
from satsom.eval.propre import PROPREClassifier


# ---------------------------------------------------
# Memory utility
# ---------------------------------------------------
def get_memory_usage():
    import psutil

    process = psutil.Process()

    mem = {}
    # CPU memory
    mem["cpu_ram_mb"] = process.memory_info().rss / (1024**2)

    # GPU memory
    if torch.cuda.is_available():
        mem["gpu_vram_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
        mem["gpu_vram_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
    else:
        mem["gpu_vram_allocated_mb"] = 0
        mem["gpu_vram_reserved_mb"] = 0

    # MPS
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        mem["mps_memory_mb"] = process.memory_info().rss / (1024**2)
    else:
        mem["mps_memory_mb"] = 0

    return mem


# ---------------------------------------------------
# Dataset enum
# ---------------------------------------------------
class EvalDataset(Enum):
    MNIST = datasets.MNIST
    FashionMNIST = datasets.FashionMNIST
    KMNIST = datasets.KMNIST


# ---------------------------------------------------
# Main memory-measure function
# ---------------------------------------------------
def eval_som(
    som_params: SatSOMParameters,
    output_path: str,
    device: str = "mps",
    save_images_each_step: bool = False,
    save_model: bool = False,
    train_perc: float = 0.8,
    size_limit: Optional[int] = None,
    eval_limit: Optional[int] = None,
    phases: list[list[int]] = ([0], [1], [2], [4], [5], [6], [7], [8], [3], [9]),
    dataset: EvalDataset = EvalDataset.FashionMNIST,
    dataset_root_dir: Optional[str] = None,
):
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # ---------------------------------------------------
    # Load dataset
    # ---------------------------------------------------
    logger.info(f"Loading {dataset.name} dataset...")
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

    # Group by class
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
    model_classes = {
        "SOM": lambda: SatSOM(som_params).to(device),
        "kNN": lambda: KNNClassifier(k=5),
        "DSDM": lambda: DSDMClassifier(
            input_dim=28 * 28,
            n_classes=len(unique_labels),
            T=2.0,
            ema=0.02,
            pruning=False,
            device=device,
        ),
        "PROPRE": lambda: PROPREClassifier(
            input_dim=28 * 28,
            nH=20,
            n_classes=len(unique_labels),
            device=device,
            lr_som=0.05,
            lr_lr=0.1,
            kappa=1.0,
            theta=0.6,
            p=10,
        ),
    }

    for model_name, model_fn in model_classes.items():
        logger.info(
            f"Running full evaluation and memory measurement for {model_name}..."
        )
        # Reset GPU memory if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = model_fn()
        mem_start = get_memory_usage()
        mem_records = []

        # ---------------------------------------------------
        # Full training/evaluation loop
        # ---------------------------------------------------
        for phase_idx, phase_labels in enumerate(phases, start=1):
            imgs_phase = torch.cat(
                [by_label["train"][lbl] for lbl in phase_labels], dim=0
            )
            labs_phase = torch.cat(
                [
                    torch.full((len(by_label["train"][lbl]),), lbl, device=device)
                    for lbl in phase_labels
                ],
                dim=0,
            )
            perm = torch.randperm(len(imgs_phase))[:size_limit]
            imgs_phase, labs_phase = imgs_phase[perm], labs_phase[perm]

            labs_oh = torch.nn.functional.one_hot(
                labs_phase, num_classes=len(unique_labels)
            )

            for i, (img, lbl_oh) in enumerate(zip(imgs_phase, labs_oh), start=1):
                if model_name == "SOM":
                    model.step(img.unsqueeze(0), lbl_oh.unsqueeze(0))
                elif model_name == "kNN":
                    model.partial_fit(img.unsqueeze(0), labs_phase[i - 1 : i])
                elif model_name == "DSDM":
                    model.partial_fit(img.unsqueeze(0), labs_phase[i - 1 : i])
                elif model_name == "PROPRE":
                    model.partial_fit(img.unsqueeze(0), labs_phase[i - 1 : i])

                # Record memory after each step
                mem_now = get_memory_usage()
                mem_diff = {k: mem_now[k] - mem_start[k] for k in mem_now}
                mem_diff.update({"model": model_name, "phase": phase_idx, "step": i})
                mem_records.append(mem_diff)

            # Save SOM images if requested
            if save_images_each_step and model_name == "SOM":
                create_satsom_image(
                    model,
                    output_path=out_dir / f"images/{model_name}_phase{phase_idx}.png",
                    img_width=28,
                    img_height=28,
                )

            # Save SOM model if requested
            if save_model and model_name == "SOM":
                state_dict_path = out_dir / f"{model_name}_phase{phase_idx}.pth"
                torch.save(model.state_dict(), state_dict_path)

        # Save memory CSV per model
        mem_df = pd.DataFrame(mem_records)
        mem_csv_path = out_dir / f"memory_curve_{model_name}.csv"
        mem_df.to_csv(mem_csv_path, index=False)
        logger.info(f"Memory curve saved at {mem_csv_path}")

        # Delete model and free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        output_path="./memory_output_full",
        size_limit=None,  # full dataset
        eval_limit=None,
        dataset=EvalDataset.FashionMNIST,
        save_images_each_step=False,
        save_model=False,
    )
