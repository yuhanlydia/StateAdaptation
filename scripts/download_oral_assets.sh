#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p artifacts/models data/raw

"$PYTHON_BIN" -m pip install 'huggingface_hub>=0.27' 'datasets>=3.0' 'wilds>=2.0' 'gdown>=5.2'

"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
    local_dir="artifacts/models/Qwen2.5-VL-7B-Instruct",
)
snapshot_download(
    "microsoft/Phi-3.5-vision-instruct",
    revision="12b77fb40b63a2c73c68243d3f767aab688a1b2a",
    local_dir="artifacts/models/Phi-3.5-vision-instruct",
)
PY

if [[ ! -d data/raw/manipbench-official/.git ]]; then
  git clone --depth 1 \
    https://github.com/slurm-lab-usc/ManipBench-Real-Robot-question.git \
    data/raw/manipbench-official
fi
git -C data/raw/manipbench-official fetch --depth 1 origin \
  39b4a3c1bd17bcc29e27993f817017040f116e04
git -C data/raw/manipbench-official checkout --detach \
  39b4a3c1bd17bcc29e27993f817017040f116e04

# Google Drive sometimes rate-limits unattended downloads. Failure is explicit;
# rerunning resumes the same zip. A browser-downloaded copy may be placed here.
if [[ ! -f data/raw/manipbench-simplified.zip ]]; then
  # This is the upstream README's alternate Simplified Dataset link. The newer
  # 1NKs... link is currently not retrievable by gdown.
  "$PYTHON_BIN" -m gdown --continue 13uhYYFYcDz9CSlCW5WKSIrgPOc3a7Z7c \
    -O data/raw/manipbench-simplified.zip || {
      echo "ManipBench automatic download was refused by Google Drive." >&2
      echo "Download the official Simplified Dataset linked in the upstream README" >&2
      echo "to data/raw/manipbench-simplified.zip, then rerun this script." >&2
      exit 3
    }
fi
mkdir -p data/raw/manipbench-simplified
"$PYTHON_BIN" - <<'PY'
import hashlib
from zipfile import ZipFile
path = "data/raw/manipbench-simplified.zip"
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
expected = "0b12f716b692ac7637dcd48ee8ae5f5e94f492eebe9fb57c2957572bcf8edf79"
if digest.hexdigest() != expected:
    raise RuntimeError(f"ManipBench zip SHA-256 mismatch: {digest.hexdigest()}")
with ZipFile(path) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"corrupt ManipBench member: {bad}")
    archive.extractall("data/raw/manipbench-simplified")
PY

# We intentionally materialize only the registered medical support/query subset.
# The official WILDS CodaLab archive is 10.66 GB and is currently unreliable.
for seed in 0 1 2; do
  "$PYTHON_BIN" scripts/prepare_camelyon17_generalization.py --seed "$seed"
  "$PYTHON_BIN" scripts/prepare_manipbench_generalization.py --seed "$seed"
done
