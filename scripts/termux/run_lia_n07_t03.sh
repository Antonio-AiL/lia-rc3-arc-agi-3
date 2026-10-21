#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v python >/dev/null 2>&1; then
  echo "Python is required. In Termux: pkg install python"
  exit 2
fi

python - <<'PY'
try:
    import numpy  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "numpy is required for the RC3 integration test. "
        "Install it in Termux before running T03."
    ) from exc
PY

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -p 'test_lia_n07_semantic.py' -v
