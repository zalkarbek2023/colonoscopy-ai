#!/usr/bin/env bash
# Fetch REAL-Colon videos one after another with the parallel downloader, unpack, drop archives.
# Usage: scripts/realcolon_queue.sh 001-007 002-009 001-001 001-006
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/data/realcolon" && mkdir -p archives frames

for id in "$@"; do
  if [ -f "frames/$id/.done" ]; then echo "$(date +%T) $id already unpacked"; continue; fi
  arc="archives/${id}_frames.tar.gz"
  # another pget may already be fetching this file: wait for it instead of starting a second copy
  while pgrep -f "[p]get.py .*${id}_frames" >/dev/null; do sleep 30; done
  if [ ! -f "$arc" ]; then
    url=$(python3 -c "import json,sys; print(next(f['download_url'] for f in json.load(open('meta/article.json'))['files'] if f['name']==sys.argv[1]))" "${id}_frames.tar.gz")
    echo "$(date +%T) downloading $id"
    python3 "$root/scripts/pget.py" "$url" "$arc" --parts 16 || { echo "$(date +%T) FAILED $id"; continue; }
  fi
  echo "$(date +%T) unpacking $id"
  mkdir -p "frames/$id" && tar -xzf "$arc" -C "frames/$id" && touch "frames/$id/.done" && rm -f "$arc"
  echo "$(date +%T) done $id ($(du -sh "frames/$id" | cut -f1))"
done
echo "$(date +%T) ALL DONE"
