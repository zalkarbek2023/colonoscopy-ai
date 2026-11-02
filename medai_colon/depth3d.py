"""Colonoscopy in 3D: polyp detection + single-frame depth -> data for a 3D point-cloud video.

For a time window of one REAL-Colon video, every `--stride`-th frame:
- crop the field of view, run the polyp detector (TensorRT engine) and Depth Anything V2 Small
  (relative inverse depth from one RGB frame, Apache-2.0),
- keep a small RGB image, the depth and the detection box for the 3D scenes,
- write 720p clips of the frames with boxes and of the depth (inferno: bright = near), plus full-size
  stills of the frame with the most confident detection.
Depth is relative (no scale), so the 3D view is a visualisation, not a measurement. Research demo.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

from prepare_yolo_v2 import fov_crop, realcolon_frames

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
GREEN = (113, 204, 46)  # BGR


def colorize(disp, lo, hi):
    n = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../models/polyp_v2.engine")
    ap.add_argument("--video", default="001-001")
    ap.add_argument("--realcolon", default="../data/realcolon")
    ap.add_argument("--start-s", type=float, default=760)
    ap.add_argument("--dur-s", type=float, default=65)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--width", type=int, default=192, help="width of the per-frame arrays for 3D")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", default="../outputs/colon3d")
    args = ap.parse_args()

    root, out = Path(args.realcolon), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fps = {r["unique_video_name"]: float(r["fps"]) for r in csv.DictReader(open(root / "meta/video_info.csv"))}[args.video]
    frames = sorted(realcolon_frames(root, args.video), key=lambda f: f[1])
    x1, y1, x2, y2 = fov_crop(frames)
    sel = [f for f in frames if args.start_s * fps <= f[1] < (args.start_s + args.dur_s) * fps][::args.stride]

    det = YOLO(args.weights, task="detect")
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, torch_dtype=torch.float16).to("cuda").eval()

    W, H = x2 - x1, y2 - y1
    w = args.width
    h = round(w * H / W)
    vh = 720
    vw = round(vh * W / H) // 2 * 2
    clip_fps = fps / args.stride
    writers = {k: cv2.VideoWriter(str(out / f"{k}.raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), clip_fps, (vw, vh))
               for k in ("rgb_box", "depth")}

    rgbs, disps, boxes, confs, ts, det_ms, depth_ms = [], [], [], [], [], [], []
    lo_s = hi_s = None
    for path, idx, _, _ in sel:
        img = cv2.imread(str(path))[y1:y2, x1:x2]
        t0 = time.perf_counter()
        r = det.predict(img, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False)[0]
        det_ms.append((time.perf_counter() - t0) * 1000)
        best = None
        if len(r.boxes):
            i = int(r.boxes.conf.argmax())
            best = (r.boxes.xyxy[i].tolist(), float(r.boxes.conf[i]))

        t0 = time.perf_counter()
        with torch.no_grad():
            px = proc(images=np.ascontiguousarray(img[:, :, ::-1]), return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
            pred = net(pixel_values=px).predicted_depth
            disp = torch.nn.functional.interpolate(pred[:, None].float(), size=(H, W), mode="bicubic",
                                                   align_corners=False)[0, 0].cpu().numpy()
        torch.cuda.synchronize()
        depth_ms.append((time.perf_counter() - t0) * 1000)

        # smoothed robust range, so the depth scale does not flicker between frames
        lo, hi = np.percentile(disp, [2, 98])
        lo_s, hi_s = (lo, hi) if lo_s is None else (0.9 * lo_s + 0.1 * lo, 0.9 * hi_s + 0.1 * hi)

        rgbs.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        disps.append(cv2.resize(disp, (w, h), interpolation=cv2.INTER_AREA).astype(np.float16))
        boxes.append(np.array(best[0]) * w / W if best else np.full(4, np.nan))
        confs.append(best[1] if best else 0.0)
        ts.append(idx / fps)

        frame = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA)
        if best:
            bx = (np.array(best[0]) * vw / W).astype(int)
            cv2.rectangle(frame, tuple(bx[:2]), tuple(bx[2:]), GREEN, 3)
        writers["rgb_box"].write(frame)
        writers["depth"].write(cv2.resize(colorize(disp, lo_s, hi_s), (vw, vh)))

    for v in writers.values():
        v.release()
    confs = np.array(confs)
    np.savez_compressed(out / "frames.npz", rgb=np.stack(rgbs), disp=np.stack(disps), box=np.stack(boxes),
                        conf=confs, t=np.array(ts), fov=np.array([x1, y1, x2, y2]))

    hero = int(confs.argmax())
    path = sel[hero][0]
    img = cv2.imread(str(path))[y1:y2, x1:x2]
    cv2.imwrite(str(out / "hero.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with torch.no_grad():
        px = proc(images=np.ascontiguousarray(img[:, :, ::-1]), return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
        pred = net(pixel_values=px).predicted_depth
        disp = torch.nn.functional.interpolate(pred[:, None].float(), size=(H, W), mode="bicubic",
                                               align_corners=False)[0, 0].cpu().numpy()
    cv2.imwrite(str(out / "hero_depth.jpg"), colorize(disp, *np.percentile(disp, [2, 98])), [cv2.IMWRITE_JPEG_QUALITY, 95])
    boxed = img.copy()
    b = (np.array(boxes[hero]) * W / w).astype(int)
    cv2.rectangle(boxed, tuple(b[:2]), tuple(b[2:]), GREEN, 4)
    cv2.imwrite(str(out / "hero_box.jpg"), boxed, [cv2.IMWRITE_JPEG_QUALITY, 95])

    meta = {"video": args.video, "fps": fps, "stride": args.stride, "fov": [x1, y1, x2, y2], "size": [w, h],
            "n": len(sel), "hero": hero, "hero_t": ts[hero], "hero_conf": float(confs[hero]),
            "det_ms_median": float(np.median(det_ms[5:])), "depth_ms_median": float(np.median(depth_ms[5:])),
            "gpu": torch.cuda.get_device_name()}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta))
    # detection timeline, one line per second: share of frames with a polyp and the max confidence
    t = np.array(ts)
    for s in range(int(t.min()), int(t.max()) + 1):
        m = (t >= s) & (t < s + 1)
        if m.any():
            print(f"{s:5d}s  hit {np.mean(confs[m] > 0):4.0%}  max {confs[m].max():.2f}")


if __name__ == "__main__":
    main()
