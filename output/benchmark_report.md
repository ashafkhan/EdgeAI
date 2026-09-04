# Edge AI Model Optimization Benchmark Report

| Runtime / Model | FPS | Mean Latency | P95 Latency | Model Size | RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (FP32)** | 24.0 | 41.63 ms | 42.86 ms | 6.25 MB | 513.4 MB |
| **ONNX Runtime (FP32)** | 26.9 | 37.24 ms | 46.16 ms | 12.26 MB | 540.9 MB |
| **ONNX Runtime (INT8)** | 27.5 | 36.35 ms | 39.52 ms | 3.34 MB | 545.7 MB |

### Optimization Highlights
- **Storage Reduction**: INT8 quantization reduces model size by **46.5%** (from 6.25 MB to 3.34 MB).
- **Throughput Factor**: 27.5 FPS vs 24.0 FPS (1.15x speedup).
