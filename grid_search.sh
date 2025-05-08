#!/usr/bin/env bash
#
#SBATCH --job-name=satsom-sweep
#SBATCH --output=logs/satsom_%A_%a.out
#SBATCH --error=logs/satsom_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5759

# This is a scrip to perform a grid hyperparameter search on a supercomputer
# Make sure to first run run_satsom.py for both datasets to download them

# --- define grid lists ---
grid_shapes=(50 100 250)              # 3
initial_lrs=(1.0 0.8 0.6 0.4 0.2)     # 5
lrs=(0.1 0.01 0.001)                  # 3
lr_biases=(1.0 0.5 0.2 0.0)           # 4
lr_sigmas=(1.0 0.1 0.01 0.001)        # 4
qs=(0.01 0.001 0.0001 0.00001)        # 4
datasets=(FashionMNIST KMNIST)        # 2

# total combinations = 3*5*3*4*4*4*2 = 5760

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

nlr=${#lrs[@]}
idx=$(( i % nlr ));                         Lr=${lrs[idx]}
i=$(( i / nlr ))

nilr=${#initial_lrs[@]}
idx=$(( i % nilr ));                        initial_lr=${initial_lrs[idx]}
i=$(( i / nilr ))

ngs=${#grid_shapes[@]}
idx=$(( i % ngs ));                         gs=${grid_shapes[idx]}
# i=$(( i / ngs ))  # not needed further

# compute dependent params
initial_sigma=$(awk "BEGIN { printf \"%.6g\", ${gs}/2 }")

# constants
input_dim=784
output_dim=10
p=10
n=10 # how many eval runs per job
output_base="./output"
device="cuda"

# make run‐specific output dir
outdir="${output_base}/gs${gs}/ilr${initial_lr}/lr${Lr}/lrbias${Lr_bias}/lrsig${lr_sigma}/q${q}/${dataset}"
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
