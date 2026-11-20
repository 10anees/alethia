**Hardware preflight — 19 November 2026**

| Item | Observed hardware / state |
| --- | --- |
| Location | Own local computer |
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB VRAM (12 GiB) |
| NVIDIA driver | 615.78.08; `nvidia-smi` reports CUDA UMD capability 13.4 |
| CPU | AMD Ryzen 5 8400F, 6 physical cores / 12 threads |
| RAM | 16 GB DDR5 |
| Swap | 4 GiB zram; this is compressed RAM, not an additional 4 GiB of physical memory |
| Project storage | NVMe SSD Gen4 |
| Operating system | Arch Linux, x86_64; kernel `7.2.14-arch1-1` |
| Python inventory | 3.11|
| Full 50-file run duration | **Not measured — full run has not been performed** |

**Measured sample benchmark — 20 November 2026**

For a hard **36-hour total covering implementation, testing and processing**, use a faster GPU for the full run. The RTX 3060 can run this workload, but the measured FP16 processing projection is approximately **26 hours**, or **31 hours with 20% scheduling headroom**. That leaves only five hours for implementation and testing. `run.py` is still empty.

All 50 Parquet files are already downloaded and the environment is installed. Their metadata contains **1,650 videos, 128.08 hours and approximately 12,984,759 frames**, rather than exactly the brief's approximate 125 hours. The frame total is estimated from each video's duration and FPS, not an exhaustive decode. Finishing processing alone in 36 hours requires about **100.2 FPS**.

| Measured configuration | Dataset-weighted throughput | Projected full processing | With 20% headroom |
| --- | --- | --- | --- |
| FP32, batch 16 | 97.7 FPS | 36.9 hours | 44.3 hours |
| FP16, batch 32 | 139.2 FPS | 25.9 hours | 31.1 hours |

These projections replace the earlier arithmetic-only scenarios. A 10,000-resample, stratified bootstrap of clip timings gives **35.2–39.0 hours for FP32** and **23.1–29.1 hours for FP16**. These intervals describe variation in the measured sample; they do not cover every possible implementation issue or system slowdown. Applying 20% headroom to the FP16 upper interval gives approximately **35 hours for processing alone**.

**What was measured**

- Inventoried metadata from all 50 files without running inference on the complete dataset.
- Selected **24 videos across 19 shards**: 16 at 640×360, four portrait, and four other formats. Selection used seed 20261120 and probability proportional to estimated video frame count within each group. Each clip starts at a random position in its video.
- Decoded and inferred **every frame in 20-second clips**: **13,652 frames / eight minutes of source footage**, processed once in each precision. No frame sampling or skipping was used inside those windows.
- Timed video opening/seeking, CPU decoding, rotation handling, BGR conversion, Ultralytics preprocessing, GPU inference and NMS, transfer of labels, per-second object presence / median people counts, and JSON Lines serialization and output writes. CUDA was synchronized around measurements.
- Measured Parquet video-column reads separately: **13.68 seconds across the 19 selected shards**, projecting about 36 seconds for 50 shards. This was added once to each full-run projection. Selected video bytes and sample output were held in temporary storage; the whole dataset was not copied.
- Weighted each format's measured seconds per frame by its estimated full-dataset frame count. An unweighted clip average would overrepresent portrait and unusual formats. Video metadata accounted for approximately 89.25% landscape 640×360 frames, 8.04% other frames and 2.71% portrait frames.
- Tuned FP32 batches 1/8/16 and FP16 batches 8/16/32 on an eight-second clip before measuring the full sample. Model warmup was outside the clip timers; new shapes encountered during the sample remained inside them.

The benchmark used **official YOLO11l weights**, image size 640, confidence 0.4, class-aware NMS at IoU 0.7, and aspect-preserving rectangular letterboxing. Other prediction defaults were retained. Software: Python 3.11.18, PyTorch 2.14.1+cu126, Ultralytics 8.4.159, PyAV 16.0.1. CPU settings were four PyTorch threads, one OpenCV thread and two video decoder threads, with cuDNN autotuning enabled. No TensorRT, model training or dependency installation was performed.

Comparing FP16 with batched FP32 on the **456 completely covered sample seconds** gave object micro-F1 **0.9970**, people-count exact agreement **100%**, and people counts within one **100%**. Partial boundary seconds were excluded. This checks precision changes on the sample; it is **not** correctness validation against the grader's complete reference or a completed submission.

Three additional **60-second FP16 clips** processed **4,739 frames in 33.75 seconds**, sustaining **131.9, 136.4 and 151.9 FPS**. These deliberately revisit two slower landscape videos and a median-throughput one at different positions; they support keeping a substantial scheduling margin rather than extrapolating the fastest tuning result. They are not an additional unbiased sample and were not substituted into the weighted projection.

Raw measurements, selection and calculation details are in [benchmark_results.json](benchmark_results.json). The temporary benchmark scripts live under `/tmp`; no submission code was written or changed. The sample results are not `results.jsonl` for submission.

**Decision for the 36-hour total**

With a separate planning allowance of **12–24 hours for implementation and correctness testing**, the central processing estimate plus scheduling headroom gives approximately **38–55 elapsed hours** if those stages run sequentially. Engineering time is an allowance, not a hardware measurement. Development can overlap an early run, but a rerun after a correctness fix can consume most of another day.

Local execution is plausible if implementation and testing are finished very quickly and the final pipeline sustains the measured FP16 throughput. It is not a comfortable choice for a firm 36-hour end-to-end limit. A faster machine buys time for validation and reruns; its exact runtime must also be benchmarked.

The supplied deadline is **Sunday, 22 November 2026, 11:55 pm PKT (UTC+05:00)**. After the actual full run, replace the unmeasured duration in the inventory above with its measured start-to-finish wall time and confirm which machine produced the submission results.
