"""Build a YOLO polyp-detection dataset from REAL-Colon and HyperKvasir.

REAL-Colon (CC BY 4.0): one VOC xml per video frame, read straight from the annotation tarballs.
Consecutive frames are nearly identical, so positives are subsampled with a stride and a smaller
share of polyp-free frames is kept as negatives. Splits are by video, never by frame.
HyperKvasir segmented (CC BY 4.0): 1000 polyp images with boxes, added to train.
Images are symlinked, not copied.
"""

import argparse
import json
import os
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def realcolon_frames(root, vid):
    """Yield (frame_path, width, height, boxes) for every annotated frame of one video."""
    with tarfile.open(root / "annotations" / f"{vid}_annotations.tar.gz") as tar:
        for m in tar:
            if not m.name.endswith(".xml"):
                continue
            x = ET.parse(tar.extractfile(m)).getroot()
            w, h = int(x.findtext("size/width")), int(x.findtext("size/height"))
            boxes = [
                tuple(int(float(o.findtext(f"bndbox/{k}"))) for k in ("xmin", "ymin", "xmax", "ymax"))
                for o in x.iter("object")
            ]
            frame = root / "frames" / vid / f"{vid}_frames" / x.findtext("filename")
            yield frame, w, h, boxes


def write_sample(out, split, name, image, w, h, boxes):
    img_dst = out / "images" / split / f"{name}{image.suffix}"
    if not img_dst.exists():
        os.symlink(image.resolve(), img_dst)
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realcolon", default="data/realcolon")
    ap.add_argument("--hyperkvasir", default="data/hyperkvasir/segmented/segmented-images")
    ap.add_argument("--train", nargs="+", required=True, help="REAL-Colon video ids for training")
    ap.add_argument("--val", nargs="+", required=True, help="REAL-Colon video ids for validation")
    ap.add_argument("--pos-stride", type=int, default=5, help="keep every Nth frame that shows a polyp")
    ap.add_argument("--neg-stride", type=int, default=60, help="keep every Nth polyp-free frame")
    ap.add_argument("--out", default="data/yolo")
    args = ap.parse_args()

    root, out = Path(args.realcolon), Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, vids in (("train", args.train), ("val", args.val)):
        for vid in vids:
            if not (root / "frames" / vid / ".done").exists():
                print(f"skip {vid}: frames not downloaded yet")
                continue
            frames = sorted(realcolon_frames(root, vid), key=lambda f: int(f[0].stem.split("_")[-1]))
            pos = [f for f in frames if f[3]][:: args.pos_stride]
            neg = [f for f in frames if not f[3]][:: args.neg_stride]
            for frame, w, h, boxes in pos + neg:
                if frame.exists():
                    write_sample(out, split, frame.stem, frame, w, h, boxes)
            print(f"{split} {vid}: {len(pos)} positive + {len(neg)} negative frames")

    hk = Path(args.hyperkvasir)
    if hk.exists():
        meta = json.load(open(hk / "bounding-boxes.json"))
        for key, m in meta.items():
            boxes = [(b["xmin"], b["ymin"], b["xmax"], b["ymax"]) for b in m["bbox"]]
            write_sample(out, "train", f"hk_{key}", hk / "images" / f"{key}.jpg", m["width"], m["height"], boxes)
        print(f"train hyperkvasir: {len(meta)} positive images")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: polyp\n"
    )
    for split in ("train", "val"):
        print(split, "total images:", len(list((out / "images" / split).iterdir())))


if __name__ == "__main__":
    main()
