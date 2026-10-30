"""Parallel HTTP range downloader for slow per-connection hosts (figshare/S3).

Usage: python3 pget.py URL OUT [--parts 16]
Resumable: finished parts are kept as OUT.partNN and skipped on re-run.
"""

import argparse
import concurrent.futures as cf
import os
import time
import urllib.request


def resolve(url):
    # A one-byte GET instead of HEAD: presigned storage URLs are signed for GET only
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req) as r:  # follows redirects to the storage URL
        return r.geturl(), int(r.headers["Content-Range"].split("/")[1])


def fetch(url, start, end, path, attempts=8):
    # storage hosts drop long connections now and then; resume the part from where it stopped
    for attempt in range(attempts):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == end - start + 1:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as e:
            if attempt == attempts - 1:
                raise
            print(f"{os.path.basename(path)}: {e}; retry {attempt + 1}", flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--parts", type=int, default=16)
    args = ap.parse_args()

    url, size = resolve(args.url)
    step = -(-size // args.parts)
    ranges = [(i * step, min(size, (i + 1) * step) - 1, f"{args.out}.part{i:02d}") for i in range(args.parts)]
    with cf.ThreadPoolExecutor(args.parts) as ex:
        for fut in cf.as_completed([ex.submit(fetch, url, *r) for r in ranges]):
            fut.result()
    with open(args.out, "wb") as out:
        for _, _, p in ranges:
            with open(p, "rb") as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk)
            os.remove(p)
    print(f"done {args.out} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
