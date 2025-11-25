#!/usr/bin/env python3
import argparse
from pathlib import Path

# Import SatSOM Parameters
from satsom.model import SatSOMParameters

# Import the specific evaluation modules
from satsom.eval import (
    eval_online,
    eval_cifar10,
    eval_cifar100,
    eval_core50,
    measure_memory,
)
from satsom.eval.eval import EvalDataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Run SatSOM eval on CIFAR, Core50, or Standard Online streams"
    )

    # Generic argument
    p.add_argument("--n", type=int, required=True, help="How many evaluations to run")

    # -------------------------------------------------
    # SatSOMParameters
    # -------------------------------------------------
    p.add_argument(
        "--grid-shape",
        type=int,
        nargs=2,
        required=True,
        help="Width and height of the SOM grid, e.g. --grid-shape 60 60",
    )
    p.add_argument(
        "--input-dim",
        type=int,
        required=True,
        help="Input dimension. NOTE: For CIFAR/Core50 this is often overwritten by embedding size (512).",
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

    # -------------------------------------------------
    # Evaluation Logic Flags
    # -------------------------------------------------
    p.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "core50", "online", "memory"],
        required=True,
        help="Which evaluation script to run.",
    )

    # Optional: If dataset == 'online', which generic dataset to use?
    p.add_argument(
        "--online-dataset-name",
        type=str,
        choices=[e.name for e in EvalDataset],
        default="FashionMNIST",
        help="If --dataset=online, which torchvision dataset to use.",
    )

    p.add_argument(
        "--output-path",
        type=str,
        default="./output",
        help="Directory to save checkpoints & images",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default="./data",
        help="Root dir for dataset downloads or Core50 raw folders",
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
        "--save-images",
        action="store_true",
        help="Save intermediate SOM images (if supported)",
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

    base_out = Path(args.output_path)
    base_out.mkdir(parents=True, exist_ok=True)

    # Loop for N evaluations
    for run_idx in range(args.n):
        run_output_path = base_out / f"{args.dataset}_run{run_idx}"

        print(f"\n>>> Starting Run {run_idx + 1}/{args.n} for {args.dataset} <<<")

        if args.dataset == "cifar10":
            # Uses eval_cifar10.py
            eval_cifar10.eval_som(
                som_params=params,
                output_path=str(run_output_path),
                device=args.device,
                dataset_root_dir=args.dataset_root,
                train_perc=args.train_perc,
                size_limit=args.size_limit,
                eval_limit=args.eval_limit,
                epochs_per_phase=args.epochs_per_phase,
                save_model=args.save_model,
                save_images_each_step=args.save_images,
                enable_logging=args.enable_logging,
                show_progress=args.show_progress,
            )

        elif args.dataset == "cifar100":
            # Uses eval_cifar100.py
            eval_cifar100.eval_som(
                som_params=params,
                output_path=str(run_output_path),
                device=args.device,
                dataset_root_dir=args.dataset_root,
                train_perc=args.train_perc,
                size_limit=args.size_limit,
                eval_limit=args.eval_limit,
                epochs_per_phase=args.epochs_per_phase,
                save_model=args.save_model,
                save_images_each_step=args.save_images,
                enable_logging=args.enable_logging,
                show_progress=args.show_progress,
            )

        elif args.dataset == "core50":
            # Uses eval_core50.py
            # Note: Core50 takes 'core50_root' instead of dataset_root_dir
            eval_core50.eval_som(
                som_params=params,
                core50_root=args.dataset_root,
                output_path=str(run_output_path),
                device=args.device,
                save_model=args.save_model,
            )

        elif args.dataset == "online":
            # Uses standard satsom.eval.eval logic (e.g. MNIST/FashionMNIST)
            eval_online.eval_som(
                som_params=params,
                output_path=run_output_path,
                device=args.device,
                dataset=EvalDataset[args.online_dataset_name],
                dataset_root_dir=args.dataset_root,
                train_perc=args.train_perc,
                size_limit=args.size_limit,
                eval_limit=args.eval_limit,
                epochs_per_phase=args.epochs_per_phase,
                save_model=args.save_model,
                save_images_each_step=args.save_images,
                enable_logging=args.enable_logging,
                show_progress=args.show_progress,
            )

        elif args.dataset == "memory":
            measure_memory.eval_som(
                som_params=params,
                output_path=run_output_path,
                device=args.device,
                dataset=EvalDataset[args.online_dataset_name],
                dataset_root_dir=args.dataset_root,
                train_perc=args.train_perc,
                size_limit=args.size_limit,
                eval_limit=args.eval_limit,
                epochs_per_phase=args.epochs_per_phase,
                save_model=args.save_model,
                save_images_each_step=args.save_images,
                enable_logging=args.enable_logging,
                show_progress=args.show_progress,
            )


if __name__ == "__main__":
    main()
