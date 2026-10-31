"""Print when each polyp is visible in a REAL-Colon video (from the per-frame annotations)."""

import csv
import sys
from pathlib import Path

from prepare_yolo_v2 import realcolon_frames

root, vid = Path(sys.argv[1]), sys.argv[2]
fps = {r["unique_video_name"]: float(r["fps"]) for r in csv.DictReader(open(root / "meta/video_info.csv"))}[vid]
seen = {}
for _, idx, _, ids in realcolon_frames(root, vid):
    for lid in ids:
        seen.setdefault(lid, []).append(idx / fps)
for lid, ts in sorted(seen.items()):
    segs, start = [], ts[0]
    for a, b in zip(ts, ts[1:]):
        if b - a > 1.0:  # a gap of over a second = left the view
            segs.append((start, a))
            start = b
    segs.append((start, ts[-1]))
    print(lid, " ".join(f"[{a:6.1f}-{b:6.1f}s]" for a, b in segs[:8]), f"({len(segs)} visits)")
