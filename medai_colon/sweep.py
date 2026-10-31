"""Offline sweep over score threshold and alarm smoothing on a dumped evaluation (eval_lesion --dump)."""

import pickle
import sys
from collections import deque

from eval_lesion import iou


def evaluate(d, conf, k, n):
    first_seen, first_hit = {}, {}
    window, alarm_on, fa = deque(maxlen=n), False, 0
    for f in d["frames"]:
        dets = [b for b, s in zip(f["dets"], f["scores"]) if s >= conf]
        for lid in f["ids"]:
            first_seen.setdefault(lid, f["idx"])
        window.append(bool(dets))
        on = sum(window) >= k
        if on and not alarm_on and not f["gts"]:
            fa += 1
        alarm_on = on
        # a polyp counts as found once an alarm is on while a detection overlaps it
        if on:
            for i, g in enumerate(f["gts"]):
                if any(iou(b, g) >= 0.2 for b in dets):
                    first_hit.setdefault(f["ids"][i], f["idx"])
    minutes = d["n_frames"] / d["fps"] / 60
    ttd = [(first_hit[l] - first_seen[l]) / d["fps"] for l in first_hit]
    return len(first_hit), len(first_seen), fa / minutes, max(ttd) if ttd else float("nan")


d = pickle.load(open(sys.argv[1], "rb"))
print(f"{d['video']}: conf  k/n  found  FA/min  worst time-to-detection")
for conf in (0.25, 0.35, 0.45, 0.55, 0.65):
    for k, n in ((3, 5), (4, 6), (5, 8), (6, 10)):
        found, total, fa, worst = evaluate(d, conf, k, n)
        print(f"        {conf:.2f}  {k}/{n:<2d}  {found}/{total}    {fa:5.2f}   {worst:5.1f} s")
