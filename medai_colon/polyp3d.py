"""Polyps in 3D: cut a polyp out of one frame and lift it above the surrounding mucosa.

For each chosen view (lesion id + view number from polyp_views.py):
  crop    a square around the box (2x its size)
  mask    SAM 2 (Ultralytics) prompted with the box
  depth   Depth Anything V2 Small on the crop: relative inverse depth, bright = near
  lift    fit a plane to the mucosa around the polyp (robust least squares outside the mask) and keep the
          elevation above it, scaled by the crop's depth range, so every polyp uses the same relative units
Writes per polyp a texture, uint16 elevation and mask grids, and labels from REAL-Colon (size, histology,
site: expert annotations). The first polyp is also run through our detector for the "find" step.
Single-frame depth has no scale: the shapes are relative, not measurements. Research demo.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import SAM, YOLO

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def square_crop(img, box, k=2.0):
    h, w = img.shape[:2]
    side = int(min(k * max(box[2] - box[0], box[3] - box[1]), h, w))
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return img[y0:y0 + side, x0:x0 + side], (x0, y0, side)


def fit_plane(z, keep):
    """Least-squares plane over pixels in `keep`, refit after dropping the 25% largest residuals."""
    yy, xx = np.nonzero(keep)
    A = np.c_[xx, yy, np.ones_like(xx)].astype(np.float64)
    v = z[yy, xx].astype(np.float64)
    for _ in range(2):
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = np.abs(A @ c - v)
        m = r <= np.quantile(r, 0.75)
        A, v = A[m], v[m]
    gy, gx = np.mgrid[:z.shape[0], :z.shape[1]]
    return c[0] * gx + c[1] * gy + c[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks", nargs="+", help="lesion:view, e.g. 001-001_2:5 (the first one is the hero)")
    ap.add_argument("--realcolon", default="../data/realcolon")
    ap.add_argument("--views", default="../outputs/polyp3d/views.json")
    ap.add_argument("--detector", default="../models/polyp_v2.engine")
    ap.add_argument("--device", default="cuda", help="cuda, or cpu when the GPU is busy (then use the .pt detector)")
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--tex", type=int, default=768)
    ap.add_argument("--out", default="../outputs/polyp3d/web")
    args = ap.parse_args()

    root, out = Path(args.realcolon), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    views = json.load(open(args.views))
    info = {r["unique_object_id"]: r for r in csv.DictReader(open(root / "meta/lesion_info.csv"))}
    fps = {r["unique_video_name"]: float(r["fps"]) for r in csv.DictReader(open(root / "meta/video_info.csv"))}
    sam = SAM("sam2.1_b.pt")
    det = YOLO(args.detector, task="detect")
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, torch_dtype=dtype).to(args.device).eval()
    G = args.grid
    polyps = []
    for n, pick in enumerate(args.picks):
        lid, k = pick.split(":")
        v = views[lid][int(k)]
        img = cv2.imread(v["path"])
        box = v["box"]
        found = None
        if n == 0:  # the hero: our detector on the full frame
            r = det.predict(img, imgsz=960, conf=0.25, device=0 if args.device == "cuda" else "cpu", verbose=False)[0]
            if len(r.boxes):
                i = int(r.boxes.conf.argmax())
                found = {"box": [round(float(x), 1) for x in r.boxes.xyxy[i].tolist()], "conf": round(float(r.boxes.conf[i]), 3)}
            cv2.imwrite(str(out / "hero_frame.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        crop, (x0, y0, side) = square_crop(img, box)
        cb = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
        mask = sam(crop, bboxes=[cb], device=args.device, verbose=False)[0].masks.data[0].cpu().numpy().astype(bool)
        mask = cv2.resize(mask.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST).astype(bool)
        with torch.no_grad():
            px = proc(images=np.ascontiguousarray(crop[:, :, ::-1]), return_tensors="pt")["pixel_values"].to(args.device, dtype)
            pred = net(pixel_values=px).predicted_depth
            disp = torch.nn.functional.interpolate(pred[:, None].float(), size=(side, side), mode="bicubic",
                                                   align_corners=False)[0, 0].cpu().numpy()
        ring = ~cv2.dilate(mask.astype(np.uint8), np.ones((side // 25 + 1,) * 2, np.uint8)).astype(bool)
        lo, hi = np.percentile(disp, [1, 99])
        elev = (disp - fit_plane(disp, ring)) / max(hi - lo, 1e-6)
        eg = cv2.resize(elev, (G, G), interpolation=cv2.INTER_AREA)
        mg = cv2.resize(mask.astype(np.float32), (G, G), interpolation=cv2.INTER_AREA)
        e0, e1 = float(eg.min()), float(eg.max())
        ((eg - e0) / (e1 - e0) * 65535).astype("<u2").tofile(out / f"p{n}.elev.u16")
        (mg * 255).astype(np.uint8).tofile(out / f"p{n}.mask.u8")
        cv2.imwrite(str(out / f"p{n}.jpg"), cv2.resize(crop, (args.tex, args.tex), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        lo_e, hi_e = np.percentile(disp, [2, 98])
        depth_vis = cv2.applyColorMap((np.clip((disp - lo_e) / (hi_e - lo_e), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out / f"p{n}_depth.jpg"), cv2.resize(depth_vis, (args.tex, args.tex)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        overlay = crop.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([113, 204, 46])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (160, 255, 190), max(2, side // 200))
        cv2.imwrite(str(out / f"p{n}_mask.jpg"), cv2.resize(overlay, (args.tex, args.tex)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = info[lid]
        inside = elev[mask]
        polyps.append({
            "id": lid, "video": r["unique_video_name"], "t": round(v["idx"] / fps[r["unique_video_name"]], 1),
            "size_mm": int(r["size [mm]"]), "histology": r["histology_extended"], "class": r["histology_class"],
            "site": r["site"], "elev_range": [round(e0, 4), round(e1, 4)], "crop": [x0, y0, side],
            "box_crop": [round(c / side, 4) for c in cb], "mask_area": round(float(mask.mean()), 3),
            "peak": round(float(np.percentile(inside, 95)), 3) if inside.size else 0.0, "found": found,
        })
        print(json.dumps(polyps[-1]))
    json.dump({"grid": G, "polyps": polyps, "frame_size": list(cv2.imread(str(out / "hero_frame.jpg")).shape[1::-1])},
              open(out / "polyps.json", "w"), indent=1)


if __name__ == "__main__":
    main()
