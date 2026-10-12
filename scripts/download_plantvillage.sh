#!/usr/bin/env bash
# Downloads the PlantVillage colour images (54,305 images, 38 classes, ~1.7 GB)
# from the official GitHub mirror using a sparse, blob-filtered clone.
set -euo pipefail
DEST=${1:-data/plantvillage}
git -c gc.auto=0 clone --depth 1 --filter=blob:none --sparse \
  https://github.com/spMohanty/PlantVillage-Dataset.git "$DEST"
cd "$DEST" && git sparse-checkout set raw/color
echo "Images in $DEST/raw/color"
