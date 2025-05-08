#!/usr/bin/env python3
import argparse
from pathlib import Path

from satsom.model import SatSOMParameters
from satsom.eval.eval import eval_som, EvalDataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Run SatSOM eval with arbitrary SatSOMParameters"
    )

    # Generic argument
    p.add_argument("--n", type=int, required=True, help="How many evaluations to run")

    # SatSOMParameters
    p.add_argument(
        "--grid-shape",
        type=int,
        nargs=2,
        required=True,
        help="Width and height of the SOM grid, e.g. --grid-shape 100 100",
    )
    p.add_argument(
        "--input-dim",
        type=int,
        required=True,
        help="Dimensionality of input vectors (e.g. 28*28)",
    )
    p.add_argument(
        "--output-dim", type=int, required=True, help="Number of output neurons/classes"
    )
    p.add_argument(
        "--initial-lr", type=float, required=True, help="Initial learning rate"
    )
    p.add_argument(
        "--initial-sigma",
        type=float,
        required=True,
        help="Initial neighborhood radius (sigma)",
    )
    p.add_argument(
        "--lr", dest="Lr", type=float, required=True, help="Global learning rate Lr"
    )
    p.add_argument(
        "--lr-bias",
        dest="Lr_bias",
        type=float,
        required=True,
        help="Bias learning rate Lr_bias",
    )
    p.add_argument(
        "--lr-sigma",
        dest="Lr_sigma",
        type=float,
        required=True,
        help="Sigma decay rate Lr_sigma",
    )
    p.add_argument(
        "--q", type=float, required=True, help="Inference constant parameter q"
    )
    p.add_argument(
        "--p", type=float, required=True, help="Inference constant parameter p"
    )

    # eval_som args
    p.add_argument(
        "--output-path",
        type=str,
        default="./output",
        help="Directory to save checkpoints & images",
    )
    p.add_argument(
        "--device", type=str, default="cuda", help="Torch device (e.g. cpu, cuda, mps)"
    )
    p.add_argument("--size-limit", type=int, default=None, help="Max samples per epoch")
    p.add_argument(
        "--eval-limit", type=int, default=None, help="Max test samples per class"
    )
    p.add_argument("--epochs-per-phase", type=int, default=1, help="Epochs per phase")
    p.add_argument(
        "--train-perc", type=float, default=0.8, help="Training fraction per class"
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=[e.name for e in EvalDataset],
        default="FashionMNIST",
        help="Dataset name (MNIST, FashionMNIST, KMNIST)",
    )
    p.add_argument(
        "--dataset-root", type=str, default=None, help="Root dir for dataset downloads"
    )
    p.add_argument(
        "--save-images", action="store_true", help="Save intermediate SOM images"
    )
    p.add_argument("--save-model", action="store_true", help="Save SOM state dicts")
    p.add_argument(
        "--no-logging",
        dest="enable_logging",
        action="store_false",
        help="Disable INFO-level logs",
    )
    p.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable tqdm bars",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build SatSOMParameters
    params = SatSOMParameters(
        grid_shape=tuple(args.grid_shape),
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        initial_lr=args.initial_lr,
        initial_sigma=args.initial_sigma,
        Lr=args.Lr,
        Lr_bias=args.Lr_bias,
        Lr_sigma=args.Lr_sigma,
        q=args.q,
        p=args.p,
    )

    # Evaluate
    for run_idx in range(args.n):
        eval_som(
            som_params=params,
            output_path=Path(args.output_path) / f"run{run_idx}",
            device=args.device,
            save_images_each_step=args.save_images,
            save_model=args.save_model,
            enable_logging=args.enable_logging,
            show_progress=args.show_progress,
            train_perc=args.train_perc,
            epochs_per_phase=args.epochs_per_phase,
            size_limit=args.size_limit,
            eval_limit=args.eval_limit,
            dataset=EvalDataset[args.dataset],
            dataset_root_dir=args.dataset_root or args.output_path,
        )


if __name__ == "__main__":
    main()
