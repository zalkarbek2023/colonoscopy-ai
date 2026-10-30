#!/usr/bin/env bash
# Download and unpack a subset of REAL-Colon (CC BY 4.0, Biffi et al., Sci Data 2024).
# Usage: scripts/download_realcolon.sh 001-007 002-009 ...
# Resumable: re-run to continue; archives are deleted after a successful unpack.
set -euo pipefail
cd "$(dirname "$0")/../data/realcolon"
mkdir -p archives frames

url_for() {
  python3 - "$1" <<'PY'
import json, sys
name = f"{sys.argv[1]}_frames.tar.gz"
files = json.load(open("meta/article.json"))["files"]
print(next(f["download_url"] for f in files if f["name"] == name))
PY
}

for id in "$@"; do
  if [ -f "frames/$id/.done" ]; then echo "$(date +%T) $id already unpacked"; continue; fi
  url=$(url_for "$id")
  echo "$(date +%T) downloading $id"
  curl -sSL -C - --retry 5 -o "archives/${id}_frames.tar.gz" "$url"
  echo "$(date +%T) unpacking $id"
  mkdir -p "frames/$id" && tar -xzf "archives/${id}_frames.tar.gz" -C "frames/$id"
  touch "frames/$id/.done" && rm -f "archives/${id}_frames.tar.gz"
  echo "$(date +%T) done $id ($(du -sh "frames/$id" | cut -f1))"
done
echo "$(date +%T) ALL DONE"
