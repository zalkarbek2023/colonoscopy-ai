"""Draw YOLO labels on a few dataset images and tile them into one sheet for a visual check."""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--prefix", default="", help="only images whose name starts with this")
    ap.add_argument("--out", default="labels_sheet.jpg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset)
    imgs = sorted(p for p in (root / "images" / args.split).iterdir() if p.name.startswith(args.prefix))
    lbl_dir = root / "labels" / args.split
    random.Random(args.seed).shuffle(imgs)
    # prefer images that do have a polyp
    imgs = [p for p in imgs if (lbl_dir / f"{p.stem}.txt").read_text().strip()][: args.n]

    tiles = []
    for p in imgs:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        for line in (lbl_dir / f"{p.stem}.txt").read_text().split("\n"):
            if not line.strip():
                continue
            _, cx, cy, bw, bh = map(float, line.split())
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        tile = cv2.resize(img, (480, int(480 * h / w)))
        tile = cv2.copyMakeBorder(tile, 0, max(0, 400 - tile.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:400]
        cv2.putText(tile, p.stem[:22], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i: i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("saved", args.out, sheet.shape)


if __name__ == "__main__":
    main()
