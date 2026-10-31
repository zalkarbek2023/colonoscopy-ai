"""Print per-epoch training losses and validation metrics of an Ultralytics run."""

import csv
import sys

rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(sys.argv[1]))]
print(f"epochs done: {len(rows)}")
print("ep  box_loss cls_loss | val P     R      mAP50  mAP50-95")
for r in rows:
    f = lambda k: float(r[k])
    print(f"{int(f('epoch')):2d}  {f('train/box_loss'):.3f}    {f('train/cls_loss'):.3f}    | "
          f"{f('metrics/precision(B)'):.3f} {f('metrics/recall(B)'):.3f}  {f('metrics/mAP50(B)'):.3f}  {f('metrics/mAP50-95(B)'):.3f}")
