"""Train the polyp detector; resumes automatically if a previous run was interrupted.

Ultralytics saves last.pt after every epoch, so a power cut costs at most one epoch:
running the same command again picks up from runs/<name>/weights/last.pt.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--name", default="polyp_v1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    # absolute project dir: a relative one gets nested under Ultralytics' own runs dir
    project = Path("runs/detect").resolve()
    last = project / args.name / "weights" / "last.pt"
    if last.exists():
        print(f"resuming {last}")
        YOLO(last).train(resume=True)
        return

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=8,
        name=args.name,
        project=str(project),
        cos_lr=True,
        patience=args.patience,
        close_mosaic=8,
        # endoscopy: the scope rotates freely; a small hue shift covers BLI/NBI light modes
        degrees=30,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.35,
        plots=True,
    )


if __name__ == "__main__":
    main()
