"""Export frames.npz from depth3d.py into small files the 3D video loads (textures, depth grids, boxes).

Scenes:
  tunnel  one frame with the open lumen, for a fly-in
  polyp   the most confident detection inside --polyp-s
  seq     every saved frame inside --seq-s, played as a live 3D clip
Depth is stored as uint16 grids of relative inverse depth scaled to 0..65535 (per frame for the stills,
one smoothed range for the clip so its scale does not flicker). Boxes are fractions of width/height.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_yolo_v2 import realcolon_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../outputs/colon3d")
    ap.add_argument("--realcolon", default="../data/realcolon")
    ap.add_argument("--tunnel-s", type=float, default=762)
    ap.add_argument("--polyp-s", type=float, nargs=2, default=(779, 783))
    ap.add_argument("--seq-s", type=float, nargs=2, default=(777, 783))
    ap.add_argument("--tex-width", type=int, default=960)
    ap.add_argument("--out", default="../outputs/colon3d/web")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    (out / "seq").mkdir(parents=True, exist_ok=True)
    meta = json.load(open(src / "meta.json"))
    d = np.load(src / "frames.npz")
    disp, box, conf, t = d["disp"].astype(np.float32), d["box"], d["conf"], d["t"]
    gw, gh = meta["size"]
    x1, y1, x2, y2 = meta["fov"]
    root = Path(args.realcolon)
    fps = {r["unique_video_name"]: float(r["fps"]) for r in csv.DictReader(open(root / "meta/video_info.csv"))}[meta["video"]]
    paths = {idx: p for p, idx, _, _ in realcolon_frames(root, meta["video"])}

    def texture(i, name):
        img = cv2.imread(str(paths[round(t[i] * fps)]))[y1:y2, x1:x2]
        h = round(args.tex_width * img.shape[0] / img.shape[1])
        cv2.imwrite(str(out / name), cv2.resize(img, (args.tex_width, h), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    def grid(a, lo, hi):
        return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 65535).astype("<u2")

    def frac_box(i):
        return None if conf[i] <= 0 else [round(float(v), 4) for v in (box[i] / [gw, gh, gw, gh])]

    scenes = {}
    for name, i in (("tunnel", int(np.argmin(abs(t - args.tunnel_s)))),
                    ("polyp", int(np.where((t >= args.polyp_s[0]) & (t < args.polyp_s[1]), conf, -1).argmax()))):
        texture(i, f"{name}.jpg")
        grid(disp[i], *np.percentile(disp[i], [1, 99])).tofile(out / f"{name}.u16")
        scenes[name] = {"t": round(float(t[i]), 2), "conf": round(float(conf[i]), 3), "box": frac_box(i)}

    seq = np.flatnonzero((t >= args.seq_s[0]) & (t < args.seq_s[1]))
    lo, hi = np.percentile(disp[seq], [1, 99])
    np.concatenate([grid(disp[i], lo, hi).ravel() for i in seq]).tofile(out / "seq.u16")
    for k, i in enumerate(seq):
        texture(i, f"seq/{k:03d}.jpg")
    scenes["seq"] = {"n": len(seq), "fps": round(fps / meta["stride"], 3),
                     "t": [round(float(t[i]), 2) for i in seq], "conf": [round(float(conf[i]), 3) for i in seq],
                     "box": [frac_box(i) for i in seq]}

    info = {"grid": [gw, gh], "tex": [args.tex_width, round(args.tex_width * (y2 - y1) / (x2 - x1))],
            "det_ms": round(meta["det_ms_median"], 1), "depth_ms": round(meta["depth_ms_median"], 1),
            "gpu": meta["gpu"], "video": meta["video"], **scenes}
    json.dump(info, open(out / "scenes.json", "w"))
    print(json.dumps({k: v for k, v in info.items() if k != "seq"}), "seq frames:", len(seq))


if __name__ == "__main__":
    main()
