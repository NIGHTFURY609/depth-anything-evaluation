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

if [ ! -x "$PY" ]; then
  echo "could not check downloads: no Python at $PY (run scripts/setup_wsl.sh first)"
  exit 2
fi

# Prints nothing; returns 0 if valid, 1 if invalid. Only the checker's own "invalid" code (3)
# may lead to a delete: a broken environment must never remove a good 15 GB download.
is_valid() {
  local rc=0
  "$PY" "$SCRIPT_DIR/verify_rellis.py" --check-download "$1" || rc=$?
  case "$rc" in
    0) return 0 ;;
    3) return 1 ;;
    *)
      echo "could not check $1: checker exited with $rc, keeping the file"
      exit 2
      ;;
  esac
}

first_file() {
  find "$1" -maxdepth 1 -type f ! -name '*.part' ! -name '*.tmp' | head -n 1
}

fetch() {
  local name="$1" id="$2" dir="$RAW/$1" existing
  mkdir -p "$dir"
  existing="$(first_file "$dir")"
  if [ -n "$existing" ]; then
    if is_valid "$existing"; then
      echo "skip  $name (already valid)"
      return
    fi
    rm -f "$existing"
  fi
  echo "fetch $name"
  "$PY" -m gdown --continue "https://drive.google.com/uc?id=$id" -O "$dir/"
  existing="$(first_file "$dir")"
  if [ -n "$existing" ] && is_valid "$existing"; then
    return
  fi
  [ -n "$existing" ] && rm -f "$existing"
  echo "FAILED $name: download by browser into $dir/ from https://drive.google.com/file/d/$id/view"
  failed=1
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
