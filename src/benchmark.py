"""
Edge AI for Smart City Surveillance - Phase 6: Edge Optimization Benchmark
Compares PyTorch, ONNX Runtime FP32, and ONNX Runtime INT8 models.
Measures: Latency (mean, median, p95), FPS, Model Size, CPU %, and RAM (MB).
"""

import argparse
import os
from pathlib import Path
import time
import cv2
import numpy as np
import onnxruntime as ort
import psutil
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS = {
    "PyTorch (FP32)": ROOT_DIR / "models" / "yolov8n.pt",
    "ONNX Runtime (FP32)": ROOT_DIR / "models" / "yolov8n.onnx",
    "ONNX Runtime (INT8)": ROOT_DIR / "models" / "yolov8n_int8.onnx",
}

SAMPLE_IMAGE = ROOT_DIR / "test.jpg"


def preprocess_image(image_path: Path, input_size: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess image into CHW RGB float32 [0, 1] tensor."""
    img0 = cv2.imread(str(image_path))
    if img0 is None:
        raise FileNotFoundError(f"Sample image not found at {image_path}")

    img = cv2.resize(img0, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # BCHW
    return img, img0


def benchmark_pytorch(model_path: Path, input_tensor: np.ndarray, num_runs: int = 100, warmup: int = 20):
    """Benchmark Ultralytics PyTorch model."""
    import torch

    model = YOLO(str(model_path))
    tensor_torch = torch.from_numpy(input_tensor)

    # Warmup
    for _ in range(warmup):
        _ = model(tensor_torch, verbose=False)

    latencies = []
    process = psutil.Process(os.getpid())
    cpu_samples = []

    for _ in range(num_runs):
        cpu_samples.append(process.cpu_percent(interval=None))
        t0 = time.perf_counter()
        _ = model(tensor_torch, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mem_info = process.memory_info().rss / (1024 * 1024)
    avg_cpu = float(np.mean([c for c in cpu_samples if c > 0] or [0.0]))

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / float(np.mean(latencies)),
        "ram_mb": mem_info,
        "cpu_pct": avg_cpu,
        "file_size_mb": model_path.stat().st_size / (1024 * 1024),
    }


def benchmark_onnx(model_path: Path, input_tensor: np.ndarray, num_runs: int = 100, warmup: int = 20):
    """Benchmark ONNX Runtime session."""
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = psutil.cpu_count(logical=False) or 4

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: input_tensor})

    latencies = []
    process = psutil.Process(os.getpid())
    cpu_samples = []

    for _ in range(num_runs):
        cpu_samples.append(process.cpu_percent(interval=None))
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: input_tensor})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mem_info = process.memory_info().rss / (1024 * 1024)
    avg_cpu = float(np.mean([c for c in cpu_samples if c > 0] or [0.0]))

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / float(np.mean(latencies)),
        "ram_mb": mem_info,
        "cpu_pct": avg_cpu,
        "file_size_mb": model_path.stat().st_size / (1024 * 1024),
    }


def run_benchmark(num_runs: int = 80):
    print("=" * 80)
    print("PHASE 6: EDGE AI MODEL OPTIMIZATION & RUNTIME BENCHMARK")
    print("=" * 80)
    print(f"Device           : Apple Silicon (CPU / CoreML / Metal)")
    print(f"Test Image       : {SAMPLE_IMAGE}")
    print(f"Benchmark Rounds : {num_runs} inferences per runtime")
    print("=" * 80 + "\n")

    input_tensor, _ = preprocess_image(SAMPLE_IMAGE)
    results = {}

    for label, path in MODELS.items():
        if not path.is_file():
            print(f"Skipping {label}: file not found at {path}")
            continue

        print(f"Running benchmark for [{label}] ({path.name})...")
        if "PyTorch" in label:
            stats = benchmark_pytorch(path, input_tensor, num_runs=num_runs)
        else:
            stats = benchmark_onnx(path, input_tensor, num_runs=num_runs)

        results[label] = stats
        print(f"  -> Mean Latency: {stats['mean_latency_ms']:.2f} ms | FPS: {stats['fps']:.1f} | Size: {stats['file_size_mb']:.2f} MB")

    # Format Markdown and Console Table
    print("\n" + "=" * 80)
    print("EDGE AI BENCHMARK RESULTS COMPARISON")
    print("=" * 80)
    header = f"{'Runtime / Model':<22} | {'FPS':<7} | {'Mean Latency':<13} | {'P95 Latency':<12} | {'Model Size':<10} | {'RAM (MB)':<9}"
    print(header)
    print("-" * len(header))

    md_table = []
    md_table.append("| Runtime / Model | FPS | Mean Latency | P95 Latency | Model Size | RAM (MB) |")
    md_table.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for label, r in results.items():
        row = (
            f"{label:<22} | "
            f"{r['fps']:5.1f} | "
            f"{r['mean_latency_ms']:8.2f} ms | "
            f"{r['p95_latency_ms']:7.2f} ms | "
            f"{r['file_size_mb']:6.2f} MB | "
            f"{r['ram_mb']:6.1f} MB"
        )
        print(row)
        md_table.append(
            f"| **{label}** | {r['fps']:.1f} | {r['mean_latency_ms']:.2f} ms | {r['p95_latency_ms']:.2f} ms | {r['file_size_mb']:.2f} MB | {r['ram_mb']:.1f} MB |"
        )

    print("=" * 80 + "\n")

    # Save to output/benchmark_report.md
    report_path = ROOT_DIR / "output" / "benchmark_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Edge AI Model Optimization Benchmark Report\n\n")
        f.write("\n".join(md_table))
        f.write("\n\n### Optimization Highlights\n")
        if "PyTorch (FP32)" in results and "ONNX Runtime (INT8)" in results:
            pt = results["PyTorch (FP32)"]
            int8 = results["ONNX Runtime (INT8)"]
            size_red = (1 - int8["file_size_mb"] / pt["file_size_mb"]) * 100
            f.write(f"- **Storage Reduction**: INT8 quantization reduces model size by **{size_red:.1f}%** (from {pt['file_size_mb']:.2f} MB to {int8['file_size_mb']:.2f} MB).\n")
            speedup = int8["fps"] / max(pt["fps"], 1e-6)
            f.write(f"- **Throughput Factor**: {int8['fps']:.1f} FPS vs {pt['fps']:.1f} FPS ({speedup:.2f}x speedup).\n")
    print(f"Benchmark report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge AI Model Benchmark")
    parser.add_argument("--runs", type=int, default=60, help="Number of benchmark iterations")
    args = parser.parse_args()
    run_benchmark(num_runs=args.runs)
