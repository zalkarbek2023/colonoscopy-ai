#!/usr/bin/env bash
# Download HyperKvasir (CC BY 4.0, Borgli et al., Sci Data 2020) from the Hugging Face mirror.
# Usage: scripts/download_hyperkvasir.sh segmented labeled videos
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/hyperkvasir && cd data/hyperkvasir
base="https://huggingface.co/datasets/SimulaMet-HOST/HyperKvasir/resolve/main"
for part in "$@"; do
  zip="hyper-kvasir-${part}-images.zip"; [ "$part" = videos ] && zip="hyper-kvasir-videos.zip"
  if [ -f "$part/.done" ]; then echo "$(date +%T) $part already unpacked"; continue; fi
  echo "$(date +%T) downloading $zip"
  curl -sSL -C - --retry 5 -o "$zip" "$base/$zip"
  echo "$(date +%T) unpacking $zip"
  mkdir -p "$part" && python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip" "$part"
  touch "$part/.done" && rm -f "$zip"
  echo "$(date +%T) done $part ($(du -sh "$part" | cut -f1))"
done
echo "$(date +%T) ALL DONE"
