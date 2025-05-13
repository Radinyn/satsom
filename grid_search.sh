#!/usr/bin/env bash
#
#SBATCH --job-name=satsom-sweep
#SBATCH --output=logs/satsom_%A_%a.out
#SBATCH --error=logs/satsom_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --account=plgccbmc13-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --array=0-383

# This is a scrip to perform a grid hyperparameter search on a supercomputer
# Make sure to first run run_satsom.py for both datasets to download them

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set venv path
VENV_DIR="$SCRIPT_DIR/venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"

    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
    if [ -f "$REQUIREMENTS" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install --upgrade pip
        pip install -r "$REQUIREMENTS"
    else
        echo "Warning: requirements.txt not found at $REQUIREMENTS"
    fi
else
    echo "Virtual environment already exists."
    source "$VENV_DIR/bin/activate"
fi

# --- define grid lists ---
grid_shapes=(50 100 250)              # 3
lr_biases=(1.0 0.5 0.2 0.0)           # 4
lr_sigmas=(1.0 0.1 0.01 0.001)        # 4
qs=(0.01 0.001 0.0001 0.00001)        # 4
datasets=(FashionMNIST KMNIST)        # 2

# total combinations = 3*4*4*4*2 = 384

# map the array task ID to each dimension
i=$SLURM_ARRAY_TASK_ID

# dataset (fastest varying)
nd=${#datasets[@]}
idx=$(( i % nd ));                          dataset=${datasets[idx]}
i=$(( i / nd ))

nq=${#qs[@]}
idx=$(( i % nq ));                          q=${qs[idx]}
i=$(( i / nq ))

nls=${#lr_sigmas[@]}
idx=$(( i % nls ));                         lr_sigma=${lr_sigmas[idx]}
i=$(( i / nls ))

nlb=${#lr_biases[@]}
idx=$(( i % nlb ));                         Lr_bias=${lr_biases[idx]}
i=$(( i / nlb ))

ngs=${#grid_shapes[@]}
idx=$(( i % ngs ));                         gs=${grid_shapes[idx]}
# i=$(( i / ngs ))  # not needed further

# compute dependent params
initial_sigma=$(awk "BEGIN { printf \"%.6g\", ${gs}/2 }")

# constants
input_dim=784
output_dim=10
initial_lr=0.5
Lr=0.01
p=10
n=10 # how many eval runs per job
output_base="./output"
device="cuda"

# make run‐specific output dir
outdir="${output_base}/gs${gs}/lrbias${Lr_bias}/lrsig${lr_sigma}/q${q}/${dataset}"
mkdir -p "$outdir"

# launch
python3 run_satsom.py \
  --n ${n} \
  --grid-shape ${gs} ${gs} \
  --input-dim ${input_dim} \
  --output-dim ${output_dim} \
  --initial-lr ${initial_lr} \
  --initial-sigma ${initial_sigma} \
  --lr ${Lr} \
  --lr-bias ${Lr_bias} \
  --lr-sigma ${lr_sigma} \
  --q ${q} \
  --p ${p} \
  --dataset ${dataset} \
  --output-path ${outdir} \
  --dataset-root ${output_base} \
  --device ${device} \
