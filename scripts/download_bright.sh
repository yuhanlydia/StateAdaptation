#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-labels}"
DESTINATION="${2:-data/raw/bright}"
RECORD="https://zenodo.org/records/20072020/files"

mkdir -p "${DESTINATION}"
zip_ok() {
  python3 - "$1" <<'PY'
from zipfile import ZipFile
import sys
try:
    with ZipFile(sys.argv[1]) as archive:
        bad = archive.testzip()
    raise SystemExit(0 if bad is None else 1)
except Exception:
    raise SystemExit(1)
PY
}
extract_zip() {
  python3 - "$1" "$2" <<'PY'
from zipfile import ZipFile
import sys
with ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
}
download() {
  local name="$1"
  local archive="${DESTINATION}/${name}"
  local partial="${archive}.part"
  if [[ -f "${archive}" ]] && ! zip_ok "${archive}"; then
    if [[ -f "${partial}" ]]; then
      echo "Both incomplete files exist for ${name}; keep the larger one and retry manually." >&2
      exit 1
    fi
    mv "${archive}" "${partial}"
  fi
  if [[ ! -f "${archive}" ]]; then
    curl -L --fail --show-error \
      --retry 8 --retry-all-errors --retry-delay 5 \
      --continue-at - \
      --output "${partial}" \
      "${RECORD}/${name}?download=1"
    mv "${partial}" "${archive}"
  fi
  extract_zip "${archive}" "${DESTINATION}"
}

download cvprw26_trainval_instance_labels.zip
echo "b8b7e8202856966d0f4ff0e3b3aa9b77  ${DESTINATION}/cvprw26_trainval_instance_labels.zip" | md5sum --check
if [[ "${MODE}" == "full" ]]; then
  download pre-event.zip
  download post-event.zip
  download target.zip
  echo "087db04490233e40fd5b53ea1d3b374a  ${DESTINATION}/pre-event.zip" | md5sum --check
  echo "13dbfff273e95995fee2a868388da4ea  ${DESTINATION}/post-event.zip" | md5sum --check
  echo "d7f48f686e0b01772949c1b5e56e3146  ${DESTINATION}/target.zip" | md5sum --check
elif [[ "${MODE}" != "labels" ]]; then
  echo "usage: $0 [labels|full] [destination]" >&2
  exit 2
fi
