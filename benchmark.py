"""CLI for CPU latency of exported ONNX models, at several thread counts. Needs no
dataset, so it runs on whatever machine you actually care about. Measures the network
only, not image loading or decoding + NMS.

Usage:
    python benchmark.py fp32=model_fp32.onnx int8=model_int8.onnx --threads 1 2 4
"""
import argparse
import os
import platform
import time

import numpy as np
import onnxruntime as ort


def measure_latency(model_path, threads, img_size=320, runs=100, warmup=10):
    """Latency in ms over `runs` single-image inferences. The first `warmup` ones are
    discarded: they include allocating buffers and cold caches."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    image = np.random.default_rng(0).random((1, 3, img_size, img_size), dtype=np.float32)
    times = []
    for i in range(warmup + runs):
        start = time.perf_counter()
        session.run(None, {"images": image})
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)

    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95))}


def benchmark(models, thread_counts, img_size=320, runs=100):
    """models: {name: path}. Returns one row per (model, thread count)."""
    rows = []
    for name, path in models.items():
        size_mb = os.path.getsize(path) / 1e6
        for threads in thread_counts:
            rows.append({"model": name, "size_mb": size_mb, "threads": threads,
                         **measure_latency(path, threads, img_size, runs)})
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="name=path.onnx ...")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--runs", type=int, default=100)
    return parser.parse_args()


def main(args):
    models = dict(m.split("=", 1) for m in args.models)
    print(f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
          f"onnxruntime {ort.__version__}, input {args.img_size}x{args.img_size}, {args.runs} runs")
    print(f"{'model':16s} {'size MB':>8s} {'threads':>8s} {'mean ms':>9s} {'median ms':>10s} {'p95 ms':>8s}")
    for r in benchmark(models, args.threads, args.img_size, args.runs):
        print(f"{r['model']:16s} {r['size_mb']:8.2f} {r['threads']:8d} {r['mean_ms']:9.1f} "
              f"{r['median_ms']:10.1f} {r['p95_ms']:8.1f}")


if __name__ == "__main__":
    main(parse_args())
