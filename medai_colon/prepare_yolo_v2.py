"""v2 dataset: like prepare_yolo.py, but each frame is cropped to the endoscope's field of view.

Videos differ: Olympus shows an octagon that fills the frame, Fujifilm a circle inside black
borders with on-screen text. A fixed per-video crop (from the median of sampled frames) removes
the borders, so more pixels go to the mucosa; boxes are shifted and clipped to the crop.
Cropped frames are written as JPEG (not symlinked) because the pixels change.
"""

import argparse
import csv
import json
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def realcolon_frames(root, vid):
    """[(frame_path, frame_idx, boxes, lesion_ids)] for every annotated frame of one video."""
    out = []
    with tarfile.open(root / "annotations" / f"{vid}_annotations.tar.gz") as tar:
        for m in tar:
            if not m.name.endswith(".xml"):
                continue
            x = ET.parse(tar.extractfile(m)).getroot()
            objs = list(x.iter("object"))
            boxes = [tuple(int(float(o.findtext(f"bndbox/{k}"))) for k in ("xmin", "ymin", "xmax", "ymax")) for o in objs]
            name = x.findtext("filename")
            out.append((root / "frames" / vid / f"{vid}_frames" / name, int(Path(name).stem.split("_")[-1]),
                        boxes, [o.findtext("unique_id") for o in objs]))
    return sorted(out, key=lambda f: f[1])


def select_positives(frames, fps, early_s, early_stride, late_stride):
    """Dense sampling while each lesion is first being found, sparse during and after its removal."""
    first = {}
    for _, idx, _, ids in frames:
        for lid in ids:
            first.setdefault(lid, idx)
    keep, seen = [], defaultdict(int)
    for f in frames:
        _, idx, boxes, ids = f
        if not boxes:
            continue
        take = False
        for lid in ids:
            stride = early_stride if idx - first[lid] < early_s * fps else late_stride
            take |= seen[lid] % stride == 0
            seen[lid] += 1
        if take:
            keep.append(f)
    return keep


def fov_crop(frames, n=40, thresh=25):
    """Bounding box of the bright field of view, from the median of n frames spread over the video."""
    picks = [frames[i][0] for i in np.linspace(0, len(frames) - 1, n).astype(int)]  # (path, ...) tuples
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in picks if p.exists()]
    med = np.median(np.stack(imgs), axis=0).astype(np.uint8)
    mask = med > thresh
    # rows/columns that are mostly field of view; sparse on-screen text in the borders doesn't count
    rows = np.flatnonzero(mask.mean(axis=1) > 0.2)
    cols = np.flatnonzero(mask.mean(axis=0) > 0.2)
    return int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)


def save(out, split, name, img, boxes):
    h, w = img.shape[:2]
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realcolon", default="data/realcolon")
    ap.add_argument("--hyperkvasir", default="data/hyperkvasir/segmented/segmented-images")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--early-seconds", type=float, default=15, help="detection phase after a lesion first appears")
    ap.add_argument("--early-stride", type=int, default=3)
    ap.add_argument("--late-stride", type=int, default=20)
    ap.add_argument("--neg-stride", type=int, default=80)
    ap.add_argument("--out", default="data/yolo_v2")
    args = ap.parse_args()

    root, out = Path(args.realcolon), Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    fps = {r["unique_video_name"]: float(r["fps"]) for r in csv.DictReader(open(root / "meta" / "video_info.csv"))}
    crops = {}
    for split, vids in (("train", args.train), ("val", args.val)):
        for vid in vids:
            if not (root / "frames" / vid / ".done").exists():
                print(f"skip {vid}: not downloaded")
                continue
            frames = realcolon_frames(root, vid)
            x1, y1, x2, y2 = crops[vid] = fov_crop(frames)
            pos = select_positives(frames, fps[vid], args.early_seconds, args.early_stride, args.late_stride)
            neg = [f for f in frames if not f[2]][:: args.neg_stride]
            for frame, _, boxes, _ in pos + neg:
                img = cv2.imread(str(frame))
                if img is None:
                    continue
                shifted = [(a - x1, b - y1, c - x1, d - y1) for a, b, c, d in boxes]
                save(out, split, frame.stem, img[y1:y2, x1:x2], shifted)
            print(f"{split} {vid}: crop {crops[vid]}, {len(pos)} positive + {len(neg)} negative frames", flush=True)

    hk = Path(args.hyperkvasir)
    if hk.exists():
        meta = json.load(open(hk / "bounding-boxes.json"))
        for key, m in meta.items():
            img = cv2.imread(str(hk / "images" / f"{key}.jpg"))
            if img is not None:
                save(out, "train", f"hk_{key}", img, [(b["xmin"], b["ymin"], b["xmax"], b["ymax"]) for b in m["bbox"]])
        print(f"train hyperkvasir: {len(meta)} images")

    json.dump(crops, open(out / "crops.json", "w"), indent=1)
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: polyp\n")
    for split in ("train", "val"):
        print(split, "total images:", len(list((out / "images" / split).iterdir())))


if __name__ == "__main__":
    main()
