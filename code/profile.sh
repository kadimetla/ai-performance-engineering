#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./profile.sh [tag] [command...]

Runs Nsight Compute with the shared section template and stores outputs under profiles/<tag>/.
Examples:
  ./profile.sh baseline ./build/my_binary --arg foo
  KERNEL_REGEX=".*attention_decode.*" ./profile.sh tuned python main.py

The tag is used as the output directory name. If no command is provided, ./app is executed.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TAG="${1:-baseline}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  CMD=("$@")
else
  CMD=(./app)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  REPO_ROOT="$ROOT"
else
  REPO_ROOT="$SCRIPT_DIR"
fi

OUTDIR="$REPO_ROOT/profiles/${TAG}"
mkdir -p "$OUTDIR"

NCTEMPLATE="$REPO_ROOT/ncu_template.ini"
if [[ ! -f "$NCTEMPLATE" ]]; then
  echo "error: missing $NCTEMPLATE" >&2
  exit 1
fi

KERNEL_REGEX="${KERNEL_REGEX:-.*}"
LOG_FILE="${OUTDIR}/metrics.csv"
REPORT_FILE="${OUTDIR}/report.ncu-rep"

echo "[profile] Writing outputs to $OUTDIR"
echo "[profile] Command: ${CMD[*]}"

ncu \
  --section-folder "$REPO_ROOT" \
  --import "$NCTEMPLATE" \
  --target-processes all \
  --set full \
  --sampling-interval auto \
  --cache-control none \
  --csv --page raw \
  --kernel-name-base demangled \
  --kernel-name "$KERNEL_REGEX" \
  --launch-skip 0 --launch-count 1 \
  --profile-from-start off \
  --export "$REPORT_FILE" \
  --log-file "$LOG_FILE" \
  "${CMD[@]}"

echo "[profile] Wrote:"
echo "  - $LOG_FILE"
echo "  - $REPORT_FILE"
