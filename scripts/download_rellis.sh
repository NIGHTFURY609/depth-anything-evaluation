#!/usr/bin/env bash
# Downloads the RELLIS-3D files we still need into <data_root>/raw/<name>/.
# Usage: download_rellis.sh small|large|all
# A file that fails validation (usually a Drive quota page) is deleted and reported.
set -uo pipefail

RAW="${RELLIS_RAW:-$HOME/da3ft-data/rellis/raw}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HOME/envs/da3ft/bin/python}"
WHICH="${1:-all}"
failed=0

fetch() {
  local name="$1" id="$2" dir="$RAW/$1" existing
  mkdir -p "$dir"
  existing="$(find "$dir" -maxdepth 1 -type f ! -name '*.part' ! -name '*.tmp' | head -n 1)"
  if [ -n "$existing" ] && "$PY" "$SCRIPT_DIR/verify_rellis.py" --check-download "$existing"; then
    echo "skip  $name (already valid)"
    return
  fi
  [ -n "$existing" ] && rm -f "$existing"
  echo "fetch $name"
  "$PY" -m gdown --continue "https://drive.google.com/uc?id=$id" -O "$dir/"
  existing="$(find "$dir" -maxdepth 1 -type f ! -name '*.part' ! -name '*.tmp' | head -n 1)"
  if [ -z "$existing" ] || ! "$PY" "$SCRIPT_DIR/verify_rellis.py" --check-download "$existing"; then
    [ -n "$existing" ] && rm -f "$existing"
    echo "FAILED $name: download by browser into $dir/ from https://drive.google.com/file/d/$id/view"
    failed=1
  fi
}

if [ "$WHICH" != "large" ]; then
  fetch ontology 1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6
  fetch intrinsics 1NAigZTJYocRSOTfgFBddZYnDsI_CSpwK
  fetch basler_to_ouster 19EOqWS9fDUFp4nsBrMCa69xs9LgIlS2e
  fetch image_split 1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ
  fetch lidar_split 1raQJPySyqDaHpc53KPnJVl3Bln6HlcVS
  fetch image_annotations_id 16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5
  fetch ouster_labels 12bsblHXtob60KrjV7lGXUQTdC5PhV8Er
fi
if [ "$WHICH" != "small" ]; then
  fetch ouster_scans 1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH
fi

exit "$failed"
