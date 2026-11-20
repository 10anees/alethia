"""Per-second object presence and people counts for FineVideo Parquet shards (YOLO11l)."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NAMES = ["person", "car", "dog", "laptop", "cell phone"]
CLASS_IDS = [0, 2, 16, 63, 67]  # COCO ids, same order as NAMES
BATCH = 32
WEIGHTS = Path(__file__).with_name("yolo11l.pt")


def summarize(frames):
    """frames: list of (people, mask) for one second -> (objects, people)."""
    n = len(frames)
    people = sorted(p for p, _ in frames)
    med = people[n // 2] if n % 2 else (people[n // 2 - 1] + people[n // 2]) // 2
    objects = [name for i, name in enumerate(NAMES) if 2 * sum(m[i] for _, m in frames) >= n]
    return objects, med


def second_line(key, sec, objects, people):
    return json.dumps({"video": key, "second": sec, "objects": objects, "people": people})


class Seconds:
    """Streams per-second lines; holds only the frames of the currently open second."""

    def __init__(self, key, write):
        self.key, self.write = key, write
        self.next, self.cur, self.frames = 0, None, []

    def add(self, sec, people, mask):
        if sec < self.next or (self.cur is not None and sec < self.cur):
            raise ValueError("non-monotonic frame timestamps")
        if self.cur is not None and sec != self.cur:
            self.flush()
        self.cur = sec
        self.frames.append((people, mask))

    def flush(self):
        if self.cur is None:
            return
        for s in range(self.next, self.cur):  # seconds with no frames
            self.write(second_line(self.key, s, [], 0))
        self.write(second_line(self.key, self.cur, *summarize(self.frames)))
        self.next, self.cur, self.frames = self.cur + 1, None, []


class Output:
    """Append-only results file plus fsynced journal of committed videos (crash/power-loss safe)."""

    def __init__(self, path):
        self.path, self.jpath = Path(path), Path(str(path) + ".journal")
        self.done, offset, jvalid = set(), 0, 0
        if self.jpath.exists():
            with open(self.jpath, "rb") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        break  # torn last line from a crash
                    if not line.endswith(b"\n"):
                        break
                    self.done.add(rec["video"])
                    offset, jvalid = rec["offset"], jvalid + len(line)
            if not self.path.exists() or self.path.stat().st_size < offset:
                sys.exit(f"error: {self.path} is shorter than its journal says; refusing to resume")
            os.truncate(self.jpath, jvalid)
            os.truncate(self.path, offset)  # drop lines of a video that never committed
        elif self.path.exists() and self.path.stat().st_size > 0:
            sys.exit(f"error: {self.path} exists without {self.jpath.name}; move it away first")
        self.out = open(self.path, "ab")
        self.journal = open(self.jpath, "ab")
        dfd = os.open(self.path.parent.resolve(), os.O_RDONLY)  # make the new files' names durable
        os.fsync(dfd)
        os.close(dfd)

    def write(self, line):
        self.out.write(line.encode() + b"\n")

    def commit(self, key):
        self.out.flush()
        os.fsync(self.out.fileno())
        self.journal.write(json.dumps({"video": key, "offset": self.out.tell()}).encode() + b"\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.done.add(key)


def infer(model, frames, secs, agg):
    # quantize=16 is FP16; it replaces the deprecated half=True in this Ultralytics version
    for r, sec in zip(model.predict(frames, imgsz=640, conf=0.4, iou=0.7, quantize=16, verbose=False), secs):
        cls = r.boxes.cls.int().tolist()
        agg.add(sec, cls.count(0), tuple(c in cls for c in CLASS_IDS))


def process_video(model, data, agg):
    with av.open(data) as container:
        stream = container.streams.video[0]
        stream.thread_type, stream.thread_count = "AUTO", 2
        tb, pts0 = stream.time_base, None
        frames, secs = [], []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("frame without timestamp")
            pts0 = frame.pts if pts0 is None else pts0
            img = frame.to_ndarray(format="bgr24")
            k = round(frame.rotation / 90) % 4  # rotation is counterclockwise degrees, like np.rot90
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
            frames.append(img)
            secs.append(math.floor((frame.pts - pts0) * tb))
            if len(frames) == BATCH:
                infer(model, frames, secs, agg)
                frames, secs = [], []
        if frames:
            infer(model, frames, secs, agg)
        if pts0 is None:
            raise ValueError("no decodable frames")
    agg.flush()


def read_rows(path):
    """Yields (row, mp4 value, read error), streaming one video at a time.
    If the Parquet file breaks, yields an error for every remaining row (or row 0 if unopenable)."""
    row, nrows = 0, 0
    try:
        pf = pq.ParquetFile(path, pre_buffer=False, buffer_size=1 << 20)
        nrows = pf.metadata.num_rows
        for batch in pf.iter_batches(batch_size=1, columns=["mp4"], use_threads=False):
            for value in batch.column(0):
                yield row, value, None
                row += 1
    except Exception as e:
        for r in range(row, max(nrows, row + 1)):
            yield r, None, f"could not read parquet: {e}"


def process_file(model, path, out):
    for row, value, err in read_rows(path):
        key = f"{path.name}:{row}"
        if key in out.done:
            continue
        t = time.time()
        agg = Seconds(key, out.write)
        try:
            if err:
                raise ValueError(err)
            if not value.is_valid:
                raise ValueError("missing video bytes")
            process_video(model, pa.BufferReader(value.as_buffer()), agg)
            status = f"{agg.next}s"
        except Exception as e:  # damaged video: keep written seconds, add one error line
            out.write(json.dumps({"video": key, "error": f"could not process video: {e}"[:200]}))
            status = f"error after {agg.next}s: {e}"
        out.commit(key)
        print(f"{key} {status} in {time.time() - t:.1f}s", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    out = Output(args.output)
    from ultralytics import YOLO  # imported late so a refused resume exits fast

    model = YOLO(str(WEIGHTS))
    for path in sorted(args.input.glob("*.parquet")):
        process_file(model, path, out)


if __name__ == "__main__":
    main()
