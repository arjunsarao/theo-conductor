#!/usr/bin/env bash
#SBATCH --job-name=fugu-diamond-caps-double
#SBATCH --partition=train
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -Eeuo pipefail
source scripts/lib/evaluation_env.sh
"$PYTHON_BIN" scripts/recover_fugu_diamond_caps.py --output-dir outputs/hle-diamond-fugu-caps-double-20261003
