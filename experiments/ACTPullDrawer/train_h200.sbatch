#!/usr/bin/env bash
#SBATCH --job-name=openvitac_ACT_pull_drawer100
#SBATCH --partition=gpu,gpux
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --nice=10000
#SBATCH --open-mode=append
#SBATCH --output=/data/wangzihao/outputs/openvitac_act/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/openvitac_act/slurm/%x_%j.err

set -euo pipefail
run_root="/data/wangzihao/outputs/openvitac_act/pull_drawer_100_seed0"
source_root="${ACT_SOURCE_ROOT:-${run_root}/source}"
python_bin="/public/home/wangzihao/.venvs/openvitac-act/bin/python"
dataset_root="/data/wangzihao/datasets/openvitac/act_pull_drawer_100_official_v1"
mkdir -p "$run_root"
exec 9>"${run_root}/training.lock"
flock -n 9 || { echo 'This ACT run already has a trainer.'; exit 1; }
cd "$source_root"
export TORCH_HOME=/data/wangzihao/outputs/openvitac_act/torch_cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export ACT_WORKER_AFFINITY_DIR="${run_root}/worker_affinity"
export ACT_STOP_FILE="${run_root}/stop_${SLURM_JOB_ID}"
rm -f "$ACT_STOP_FILE"
trap 'printf SIGUSR1 > "$ACT_STOP_FILE"' USR1
trap 'printf SIGTERM > "$ACT_STOP_FILE"' TERM INT
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
"$python_bin" - <<'PY'
from pathlib import Path
import hashlib,json
root=Path.cwd(); manifest=json.loads((root/'SOURCE_SHA256.json').read_text())
for name,expected in manifest.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Frozen ACT source changed: '+name)
print('Frozen source integrity passed:',len(manifest),'files')
PY
if [[ ! -f "${run_root}/COMPLETE.json" ]]; then
    srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
        "$python_bin" -u train_baseline.py --dataset "$dataset_root" \
        --output "$run_root" --device cuda:0 --workers 6 &
    step_pid=$!
    result=0
    while kill -0 "$step_pid" 2>/dev/null; do
        wait "$step_pid" || result=$?
    done
    if [[ "$result" != 0 && "$result" -lt 128 ]]; then exit "$result"; fi
fi
if [[ -f "${run_root}/COMPLETE.json" ]]; then
    if [[ ! -f "${run_root}/handoff/final/ACTPullDrawer.tar.gz" ]]; then
        "$python_bin" export_handoff.py --run "$run_root" --source "$source_root" \
            --output "${run_root}/handoff/final"
    fi
elif [[ -f "$ACT_STOP_FILE" && -f "${run_root}/checkpoints/training_latest.pt" ]]; then
    if [[ "$(cat "$ACT_STOP_FILE")" == SIGUSR1 ]]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
else
    echo 'ACT did not complete or stop at a recoverable time-limit boundary.' >&2
    exit 1
fi
