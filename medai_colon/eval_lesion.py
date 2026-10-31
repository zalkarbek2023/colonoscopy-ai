"""Clinician-level evaluation on a full held-out REAL-Colon video.

Frame metrics alone (mAP) say little about an assistant. This reports, per video:
- lesion sensitivity: share of polyps the model flags at least once (IoU >= 0.2 with the polyp box);
- time to detection: seconds from a polyp's first appearance to its first flagged frame;
- false alarms per minute: alarm *events* not overlapping any polyp, after the same temporal
  smoothing a live assistant would use (an alarm needs hits in k of the last n processed frames).
Frames are cropped to the field of view exactly as in training (prepare_yolo_v2.fov_crop).
"""

import argparse
import csv
import json
import pickle
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from prepare_yolo_v2 import fov_crop, realcolon_frames


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", default="001-001")
    ap.add_argument("--realcolon", default="data/realcolon")
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=3, help="alarm needs hits in k ...")
    ap.add_argument("--n", type=int, default=5, help="... of the last n processed frames")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--dump", help="also save every frame's detections (with scores) to this pickle")
    args = ap.parse_args()

    root = Path(args.realcolon)
    fps = {r["unique_video_name"]: float(r["fps"]) for r in csv.DictReader(open(root / "meta/video_info.csv"))}[args.video]
    frames = realcolon_frames(root, args.video)
    x1, y1, x2, y2 = fov_crop(frames)
    model = YOLO(args.weights)

    first_seen, first_hit = {}, {}
    window = deque(maxlen=args.n)
    alarm_on, fa_events, fp_frames, tp_frames, gt_frames, det_frames = False, 0, 0, 0, 0, 0
    per_frame, dumped = [], []
    for path, idx, boxes, ids in frames[:: args.stride]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = img[y1:y2, x1:x2]
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False, quantize=16)[0]
        dets = [tuple(b) for b in r.boxes.xyxy.cpu().numpy().tolist()]
        scores = r.boxes.conf.cpu().numpy().tolist()
        gts = [(a - x1, b - y1, c - x1, d - y1) for a, b, c, d in boxes]
        if args.dump:
            dumped.append({"idx": idx, "gts": gts, "ids": ids, "dets": dets, "scores": scores})
        for lid in ids:
            first_seen.setdefault(lid, idx)
        matched_gt = set()
        unmatched_det = 0
        for d in dets:
            hits = [i for i, g in enumerate(gts) if iou(d, g) >= 0.2]
            if hits:
                matched_gt.update(hits)
            else:
                unmatched_det += 1
        for i in matched_gt:
            first_hit.setdefault(ids[i], idx)
        gt_frames += bool(gts)
        det_frames += bool(dets)
        tp_frames += bool(gts) and bool(matched_gt)
        fp_frames += unmatched_det > 0 and not gts

        # live-assistant smoothing: an alarm event starts when k of the last n frames had a detection;
        # it is a false alarm if the frame that triggers it shows no polyp
        window.append(bool(dets))
        on = sum(window) >= args.k
        if on and not alarm_on and not gts:
            fa_events += 1
        alarm_on = on
        per_frame.append({"idx": idx, "n_gt": len(gts), "n_det": len(dets), "matched": len(matched_gt)})

    minutes = len(frames) / fps / 60
    lesions = sorted(first_seen)
    print(f"video {args.video}: {len(frames)} frames, {minutes:.1f} min, {len(lesions)} polyps, crop {(x1, y1, x2, y2)}")
    print(f"weights {args.weights}, conf {args.conf}, every {args.stride} frame(s), alarm = {args.k}/{args.n}")
    for lid in lesions:
        if lid in first_hit:
            print(f"  {lid}: FOUND after {(first_hit[lid] - first_seen[lid]) / fps:5.1f} s")
        else:
            print(f"  {lid}: MISSED")
    found = sum(l in first_hit for l in lesions)
    print(f"lesion sensitivity: {found}/{len(lesions)}")
    print(f"frame recall (polyp frames with a hit): {tp_frames / max(gt_frames, 1):.1%}")
    print(f"false alarm events: {fa_events} -> {fa_events / minutes:.2f} per minute")
    print(f"polyp-free frames with a detection: {fp_frames / max(len(per_frame) - gt_frames, 1):.2%}")
    Path(args.out).mkdir(exist_ok=True)
    if args.dump:
        pickle.dump({"video": args.video, "fps": fps, "stride": args.stride, "n_frames": len(frames),
                     "frames": dumped}, open(args.dump, "wb"))
    json.dump({"video": args.video, "conf": args.conf, "found": found, "lesions": len(lesions),
               "fa_per_min": fa_events / minutes, "frame_recall": tp_frames / max(gt_frames, 1),
               "ttd_s": {l: (first_hit[l] - first_seen[l]) / fps for l in first_hit}},
              open(Path(args.out) / f"{args.video}_conf{args.conf}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
