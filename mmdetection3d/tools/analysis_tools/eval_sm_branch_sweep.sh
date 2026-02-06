#!/usr/bin/env bash
set -euo pipefail

# Evaluate KITTI val AP for SM_branch{N} runs (epoch_40) via the official
# `tools/test.py` entrypoint.
#
# Run from mmdetection3d/:
#   bash tools/analysis_tools/eval_sm_branch_sweep.sh

BRANCHES=(1 2 3 4)
EPOCH=40

# Where training checkpoints live (relative to mmdetection3d/).
WORK_ROOT="work_dirs/RefineMoE"

# Write summary under repo-root results/ (ignored by git).
OUT_DIR="../results/ap/sm_branch_sweep_epoch${EPOCH}"
mkdir -p "${OUT_DIR}"

CSV_OUT="${OUT_DIR}/summary.csv"
JSON_OUT="${OUT_DIR}/summary.json"

tmp_jsonl="${OUT_DIR}/_rows.jsonl"
rm -f "${tmp_jsonl}"

echo "branch,log_json" >"${CSV_OUT}"

for N in "${BRANCHES[@]}"; do
  cfg="configs/RefineMoE/SM_branch${N}.py"
  ckpt="${WORK_ROOT}/SM_branch${N}/epoch_${EPOCH}.pth"
  eval_work_dir="${WORK_ROOT}/SM_branch${N}/eval_epoch${EPOCH}_ap"

  echo "[SM_branch${N}] cfg=${cfg} ckpt=${ckpt}"

  mkdir -p "${eval_work_dir}"
  log_prefix="${eval_work_dir}/test"

  # Evaluator is configured in SM_branch{N}.py (KittiMetric).
  python tools/test.py "${cfg}" "${ckpt}" \
    --work-dir "${eval_work_dir}" \
    2>&1 | tee "${log_prefix}.log"

  # Find the most recent json log in the eval work dir.
  log_json=""
  if ls "${eval_work_dir}"/*.log.json >/dev/null 2>&1; then
    log_json=$(ls -t "${eval_work_dir}"/*.log.json | head -n 1)
  fi

  if [[ -z "${log_json}" ]]; then
    echo "[SM_branch${N}] WARNING: no *.log.json found in ${eval_work_dir}" >&2
    echo "${N}," >>"${CSV_OUT}"
    continue
  fi

  echo "${N},${log_json}" >>"${CSV_OUT}"

  # Extract the last record containing KITTI metrics.
  python - "${N}" "${log_json}" >>"${tmp_jsonl}" <<'PY'
import json
import sys

branch = int(sys.argv[1])
path = sys.argv[2]

last = None
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if any(isinstance(k, str) and k.startswith('KITTI/') for k in d.keys()):
            last = d

out = {
    'branch_num': branch,
    'log_json': path,
    'metrics': {},
}
if last:
    out['metrics'] = {k: v for k, v in last.items() if isinstance(k, str) and k.startswith('KITTI/')}

print(json.dumps(out, ensure_ascii=False))
PY
done

# Convert JSONL rows to a single JSON file.
python - "${tmp_jsonl}" "${JSON_OUT}" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
rows = []
with open(src, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

with open(dst, 'w', encoding='utf-8') as f:
    json.dump({'results': rows}, f, indent=2, ensure_ascii=False)
print(f'Wrote: {dst}')
PY

echo "Wrote: ${CSV_OUT}"
echo "Wrote: ${JSON_OUT}"
