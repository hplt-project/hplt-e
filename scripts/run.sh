#!/bin/bash
#SBATCH --job-name=<YOUR JOB NAME>
#SBATCH --account=<YOUR PROJECT NAME>
#SBATCH --partition=small
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=128G
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/your_log_name.out
#SBATCH --error=logs/your_log_name.err

EBU_USER_PREFIX=/projappl/project_465001925/software/

module --quiet purge
module load LUMI
module load PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

export LC_ALL=en_US.UTF-8

export MODEL=${1}
export TASK=${2}

srun singularity exec $SIF lm_eval --model hf \
    --model_args pretrained=${MODEL} \
    --tasks ${TASK} \
    --include_path ./ \
    --output results/${TASK}/  \
    --write_out \
    --log_samples \
    --show_config \
    --num_fewshot 0 \
    --batch_size auto
