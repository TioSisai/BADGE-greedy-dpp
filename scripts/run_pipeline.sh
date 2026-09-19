#!/usr/bin/env bash
# Bind experiments to the specified GPU and its NUMA-local CPU cores.
# MPS renumbers devices, so CUDA_VISIBLE_DEVICES uses a UUID.
# Dataset, strategy, and seed combinations across processes should not overlap to avoid overwriting outputs.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PIPELINE_PY="${SCRIPT_DIR}/pipeline.py"
readonly PYTHON_BIN="${BADGE_GREEDY_DPP_PYTHON:-python}"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --cuda-id N [pipeline.py arguments...]

  --cuda-id N   GPU index (same as nvidia-smi), the process is also bound to that card's NUMA-local CPU cores.

The remaining arguments are passed through to pipeline.py.
Interpreter defaults to python in PATH, set via BADGE_GREEDY_DPP_PYTHON.

Examples:
  $(basename "${BASH_SOURCE[0]}") --cuda-id 0 --seed 0 --runs 2
  $(basename "${BASH_SOURCE[0]}") --cuda-id 1 --strategies full-supervised --n-tasks 1
EOF
}

cuda_id=""
want_help=0
declare -a pipeline_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-id)
      [[ $# -ge 2 ]] || { echo "ERROR: --cuda-id requires a GPU number" >&2; exit 1; }
      cuda_id="$2"
      shift 2
      ;;
    --cuda-id=*)
      cuda_id="${1#*=}"
      shift
      ;;
    -h|--help)
      want_help=1
      shift
      ;;
    *)
      pipeline_args+=("$1")
      shift
      ;;
  esac
done

if [[ "${want_help}" -eq 1 ]]; then
  usage
  echo
  echo "===== Below are pipeline.py's own arguments ====="
  exec "${PYTHON_BIN}" "${PIPELINE_PY}" --help
fi

if [[ -z "${cuda_id}" ]]; then
  echo "ERROR: missing --cuda-id" >&2
  echo >&2
  usage >&2
  exit 1
fi
if [[ ! "${cuda_id}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --cuda-id must be a non-negative integer, got: ${cuda_id}" >&2
  exit 1
fi

gpu_uuid="$(nvidia-smi -i "${cuda_id}" --query-gpu=uuid --format=csv,noheader 2>/dev/null)" || {
  echo "ERROR: GPU${cuda_id} does not exist (see nvidia-smi -L for available numbers)." >&2
  exit 1
}

# The first CPU list field in a topology row is CPU Affinity.
# Save the output first to prevent an early downstream exit from causing nvidia-smi to receive SIGPIPE.
topo_table="$(nvidia-smi topo -m 2>/dev/null)" || {
  echo "ERROR: nvidia-smi topo -m failed, cannot determine the affinity between GPU and CPU." >&2
  exit 1
}
cpu_list="$(awk -v gpu="GPU${cuda_id}" '
  $1 == gpu && cpus == "" {
    for (i = 2; i <= NF; i++)
      if ($i ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) { cpus = $i; break }
  }
  END { print cpus }' <<< "${topo_table}")"
if [[ -z "${cpu_list}" ]]; then
  echo "ERROR: failed to get the CPU Affinity of GPU${cuda_id} from nvidia-smi topo -m;" >&2
  echo "       confirm that this GPU number exists (nvidia-smi -L) and that nvidia-smi is available." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_uuid}"

echo "GPU${cuda_id} (${gpu_uuid}) <- CPU ${cpu_list}"

exec taskset -c "${cpu_list}" "${PYTHON_BIN}" "${PIPELINE_PY}" "${pipeline_args[@]}"
