"""Demo video: real-time polyp detection with "polyp memory" on a held-out colonoscopy.

- A detection becomes a polyp alarm only after it persists (hits in >= `--k` of the last `--n`
  frames), so single-frame flashes never reach the screen.
- Every confirmed polyp gets a number. When it leaves the view or hides behind a fold, a dashed
  "ghost" stays where it was last seen with the time since; if a polyp reappears within
  `--memory-s` seconds it keeps its number instead of being counted again.
- A side panel shows the unique-polyp count, the list, model speed and data credit.
Research demo, not a medical device.
"""

import argparse
import csv
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from prepare_yolo_v2 import fov_crop, realcolon_frames

GREEN, AMBER, WHITE, GREY, PANEL = (46, 204, 113), (245, 176, 65), (255, 255, 255), (150, 160, 170), (14, 18, 26)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def dashed_rect(draw, box, color, dash=14, width=3):
    x1, y1, x2, y2 = box
    for a, b, horiz, c in ((x1, x2, True, y1), (x1, x2, True, y2), (y1, y2, False, x1), (y1, y2, False, x2)):
        for s in range(int(a), int(b), dash * 2):
            e = min(s + dash, b)
            draw.line([(s, c), (e, c)] if horiz else [(c, s), (c, e)], fill=color, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--realcolon", default="../data/realcolon")
    ap.add_argument("--start-s", type=float, required=True)
    ap.add_argument("--dur-s", type=float, default=40)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--k", type=int, default=15, help="alarm: hits in k ...")
    ap.add_argument("--n", type=int, default=24, help="... of the last n frames (~0.8 s at 30 fps)")
    ap.add_argument("--memory-s", type=float, default=15)
    ap.add_argument("--speed-label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.realcolon)
    fps = {r["unique_video_name"]: float(r["fps"]) for r in csv.DictReader(open(root / "meta/video_info.csv"))}[args.video]
    frames = realcolon_frames(root, args.video)
    x1, y1, x2, y2 = fov_crop(frames)
    sel = [f for f in frames if args.start_s * fps <= f[1] < (args.start_s + args.dur_s) * fps]
    model = YOLO(args.weights, task="detect")

    vw, vh = x2 - x1, y2 - y1
    scale = 1080 / vh
    fw = int(vw * scale)
    W, H, PW = fw + 560, 1080, 560
    tmp = Path(args.out).with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    f_title, f_big, f_txt, f_small = (ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT_B, 64),
                                      ImageFont.truetype(FONT, 27), ImageFont.truetype(FONT, 22))

    hits = deque(maxlen=args.n)
    polyps = []  # {num, first_t, last_t, box, conf}
    active = None  # index into polyps of the polyp currently in view
    infer_ms = deque(maxlen=60)
    for fi, (path, idx, _, _) in enumerate(sel):
        t = idx / fps
        img = cv2.imread(str(path))[y1:y2, x1:x2]
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False)[0]
        infer_ms.append((time.perf_counter() - t0) * 1000)
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs)) if len(confs) else None
        hits.append(best is not None)
        confirmed = best is not None and sum(hits) >= args.k

        if confirmed:
            box, c = boxes[best], float(confs[best])
            if active is None:
                # a polyp seen recently and now out of view keeps its number (memory)
                recent = [i for i, p in enumerate(polyps) if t - p["last_t"] <= args.memory_s]
                if recent:
                    active = max(recent, key=lambda i: polyps[i]["last_t"])
                else:
                    polyps.append({"num": len(polyps) + 1, "first_t": t, "last_t": t, "box": box, "conf": c})
                    active = len(polyps) - 1
            polyps[active].update(last_t=t, box=box, conf=c)
        elif sum(hits) == 0:
            active = None

        frame = cv2.resize(img, (fw, H))
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:, :fw] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas[:, fw:] = PANEL
        pil = Image.fromarray(canvas)
        d = ImageDraw.Draw(pil, "RGBA")

        for i, p in enumerate(polyps):
            bx = [v * scale for v in p["box"]]
            if i == active and confirmed:
                d.rounded_rectangle(bx, radius=10, outline=GREEN, width=5)
                lbl = f"Polyp #{p['num']}  {p['conf']:.0%}"
                tw = d.textlength(lbl, font=f_txt)
                d.rounded_rectangle([bx[0], bx[1] - 40, bx[0] + tw + 20, bx[1] - 6], radius=8, fill=GREEN + (235,))
                d.text((bx[0] + 10, bx[1] - 37), lbl, font=f_txt, fill=(8, 20, 12))
            elif t - p["last_t"] <= args.memory_s:
                dashed_rect(d, bx, AMBER)
                d.text((bx[0], bx[3] + 6), f"#{p['num']} out of view · {t - p['last_t']:.1f} s", font=f_small, fill=AMBER)

        px = fw + 36
        d.text((px, 40), "COLONOSCOPY AI ASSISTANT", font=f_title, fill=WHITE)
        d.text((px, 80), "research demo · not a medical device", font=f_small, fill=GREY)
        d.text((px, 150), "Polyps found", font=f_txt, fill=GREY)
        d.text((px, 180), str(len(polyps)), font=f_big, fill=GREEN)
        y = 290
        for i, p in enumerate(polyps):
            state = "in view" if (i == active and confirmed) else (
                f"out of view {t - p['last_t']:.0f} s" if t - p["last_t"] <= args.memory_s else "done")
            col = GREEN if state == "in view" else (AMBER if "out" in state else GREY)
            d.ellipse([px, y + 6, px + 16, y + 22], fill=col)
            d.text((px + 28, y), f"Polyp #{p['num']} · found at {p['first_t'] - args.start_s:05.1f} s · {state}", font=f_small, fill=WHITE)
            y += 40
        d.text((px, H - 190), "YOLO26m · polyp memory", font=f_small, fill=GREY)
        d.text((px, H - 160), f"{args.speed_label or 'PyTorch'} · {np.mean(infer_ms):.0f} ms/frame", font=f_small, fill=GREY)
        d.text((px, H - 110), "Video: REAL-Colon dataset, Biffi et al.,", font=f_small, fill=GREY)
        d.text((px, H - 85), "Sci Data 2024 (CC BY 4.0), held-out", font=f_small, fill=GREY)
        d.text((px, H - 60), "video never seen in training", font=f_small, fill=GREY)
        writer.write(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
        if fi % 300 == 0:
            print(f"frame {fi}/{len(sel)}  polyps so far: {len(polyps)}", flush=True)

    writer.release()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp), "-c:v", "libx264",
                    "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], check=True)
    tmp.unlink()
    print(f"saved {args.out}; unique polyps: {len(polyps)}")


if __name__ == "__main__":
    main()
