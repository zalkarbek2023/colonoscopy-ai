"""Pick the best views of chosen polyps for 3D: large, sharp, fully inside the field of view.

For each lesion id, scores every annotated frame by box area x sharpness (variance of the Laplacian
inside the box), keeps the best frames at least 1 s apart and writes a contact sheet of crops so the
view can be chosen by eye. Boxes are the expert annotations of REAL-Colon.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_yolo_v2 import fov_crop, realcolon_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lesions", nargs="+", help="unique_object_id, e.g. 003-013_2")
    ap.add_argument("--realcolon", default="../data/realcolon")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out", default="../outputs/polyp3d")
    args = ap.parse_args()

    root, out = Path(args.realcolon), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {r["unique_object_id"]: r for r in csv.DictReader(open(root / "meta/lesion_info.csv"))}
    fps = {r["unique_video_name"]: float(r["fps"]) for r in csv.DictReader(open(root / "meta/video_info.csv"))}
    picks = {}
    by_video = {}
    for lid in args.lesions:
        by_video.setdefault(info[lid]["unique_video_name"], []).append(lid)
    for vid, lids in by_video.items():
        frames = realcolon_frames(root, vid)
        x1, y1, x2, y2 = fov_crop(frames)
        for lid in lids:
            cands = []
            for path, idx, boxes, ids in frames:
                for b, i in zip(boxes, ids):
                    if i != lid:
                        continue
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    m = 0.15 * max(bw, bh)
                    if b[0] - m < x1 or b[1] - m < y1 or b[2] + m > x2 or b[3] + m > y2:
                        continue  # cut by the edge of the view
                    cands.append((bw * bh, idx, path, b))
            cands.sort(reverse=True)
            scored = []
            for area, idx, path, b in cands[:150]:
                g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[b[1]:b[3], b[0]:b[2]]
                scored.append((area * np.sqrt(cv2.Laplacian(g, cv2.CV_64F).var()), idx, path, b))
            scored.sort(reverse=True)
            chosen = []
            for s, idx, path, b in scored:
                if all(abs(idx - c[1]) > fps[vid] for c in chosen):
                    chosen.append((s, idx, path, b))
                if len(chosen) == args.top:
                    break
            tiles = []
            for k, (s, idx, path, b) in enumerate(chosen):
                img = cv2.imread(str(path))
                side = int(1.8 * max(b[2] - b[0], b[3] - b[1]))
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                crop = img[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2].copy()
                crop = cv2.resize(crop, (300, 300))
                cv2.putText(crop, f"{k} f{idx} {idx / fps[vid]:.0f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                tiles.append(crop)
            r = info[lid]
            print(f"{lid}: {r['size [mm]']} mm {r['histology_extended']} ({r['site']}), {len(cands)} usable frames")
            if tiles:
                cv2.imwrite(str(out / f"views_{lid}.jpg"), np.hstack(tiles + [np.zeros_like(tiles[0])] * (args.top - len(tiles))))
            picks[lid] = [{"idx": int(c[1]), "path": str(c[2]), "box": [int(v) for v in c[3]]} for c in chosen]
    json.dump(picks, open(out / "views.json", "w"), indent=1)


if __name__ == "__main__":
    main()
