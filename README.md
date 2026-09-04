# Edge AI for Smart City Surveillance

An end-to-end, edge-native computer vision and video analytics system for smart-city and CCTV surveillance. The system executes local deep-learning inference, persistent multi-object tracking, zone analytics, overcrowding detection, virtual line crossing counting, and event logging directly on edge devices without relying on cloud processing.

---

## 🏛️ System Architecture

```
                                      CCTV / Video Stream
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │    Pre-processing & Scaling     │
                              │        (OpenCV Pipeline)        │
                              └────────────────┬────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │       YOLOv8n Inference         │
                              │ (PyTorch / ONNX Runtime / INT8) │
                              └────────────────┬────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │      ByteTrack Association      │
                              │   (Persistent IDs & Filtering)  │
                              └────────────────┬────────────────┘
                                               │
                 ┌─────────────────────────────┼─────────────────────────────┐
                 │                             │                             │
                 ▼                             ▼                             ▼
   ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
   │    Crowd Monitoring       │ │    Restricted Roadway     │ │   Virtual Line Crossing   │
   │  - Polygon Center Test    │ │  - Intrusion Detection    │ │  - Bidirectional Vectors  │
   │  - Person-Only Filter     │ │  - Jaywalking / Hazard    │ │  - Anti-Oscillation State │
   │  - Overcrowding Threshold │ │  - Instant Visual Alert   │ │  - In / Out Counters      │
   └─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘
                 │                             │                             │
                 └─────────────────────────────┼─────────────────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │    Surveillance Logger Engine   │
                              │  - SQLite Event Storage         │
                              │  - CSV Export & Debouncing      │
                              └────────────────┬────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │    Live CCTV HUD & Annotated    │
                              │          Video Output           │
                              └─────────────────────────────────┘
```

---

## 📋 Features & Roadmap Status

| Phase | Module | Description | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `src/phase1_test.py` | Virtual environment, YOLOv8n initialization, static image verification | **Completed** |
| **Phase 2** | `src/detect.py` | Real-time video detection, class filtering, bounding boxes, FPS | **Completed** |
| **Phase 3** | `src/tracker.py` | ByteTrack multi-object tracking with persistent IDs & visual trails | **Completed** |
| **Phase 4** | `src/zones.py` | Calibrated crowd monitoring, overcrowding alerts, restricted zone intrusion, bidirectional counting | **Completed** |
| **Phase 5** | `src/logger.py` | Structured event logging to SQLite database (`surveillance.db`) and CSV (`events.csv`) with debouncing | **Completed** |
| **Phase 6** | `src/benchmark.py` | ONNX export (`yolov8n.onnx`), INT8 dynamic quantization (`yolov8n_int8.onnx`), runtime benchmarking | **Completed** |
| **Phase 7** | Deployment Specs | Edge device deployment blueprints (Raspberry Pi 5, Jetson Orin, Intel NUC) | **Documented** |
| **Phase 8** | Config & Tooling | Zero-hardcoding YAML (`config.yaml`), interactive zone calibration tool (`calibrate_zones.py`) | **Completed** |

---

## 🚀 Benchmark & Model Optimization Results

Benchmarked on Apple Silicon (M2 Edge CPU) using 50 inference passes per runtime:

| Runtime / Model | Precision | FPS | Mean Latency | P95 Latency | Model Size | Memory (RAM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch** (`yolov8n.pt`) | FP32 | 22.7 | 44.09 ms | 44.59 ms | 6.25 MB | 513.8 MB |
| **ONNX Runtime** (`yolov8n.onnx`) | FP32 | 26.5 | 37.73 ms | 40.61 ms | 12.26 MB | 523.4 MB |
| **ONNX Runtime** (`yolov8n_int8.onnx`) | **INT8** | **27.1** | **36.93 ms** | **41.16 ms** | **3.34 MB** | **528.2 MB** |

### Key Takeaways:
- **INT8 Quantization**: Reduces model footprint by **46.6%** compared to PyTorch and **72.7%** compared to ONNX FP32 (from 12.26 MB to 3.34 MB).
- **Latency Reduction**: Mean latency drops from **44.09 ms** to **36.93 ms**, delivering **~27.1 FPS** throughput on low-power edge CPU.

---

## 📐 Surveillance Zone & Geometry Architecture

### 1. Crowd Monitoring Zone
- **Location**: Pedestrian Crosswalk corridor (`[(200, 950), (2120, 950), (2120, 1530), (200, 1530)]`).
- **Semantic Rule**: Evaluated **ONLY** on tracked objects with `class_name == "person"`.
- **Logic**: Objects outside the polygon do not contribute to crowd count. If `crowd_zone_people >= threshold` (default: 6), a high-visibility `!! OVERCROWDING ALERT !!` banner is triggered directly over the crowd zone and reflected in the HUD.

### 2. Restricted Roadway Hazard Zone
- **Location**: Active vehicle roadway intersection (`[(150, 1550), (2050, 1550), (1900, 2650), (150, 2650)]`).
- **Semantic Rule**: Monitors unauthorized intrusion into active traffic lanes. While cars/taxis/buses operate legally, any `person` or `bicycle` entering the active roadway triggers `RESTRICTED INTRUSION [ID:X]`.

### 3. Virtual Counting Line
- **Location**: Midline bisecting the crosswalk (`(1100, 950)` to `(1100, 1530)`).
- **Anti-Oscillation Algorithm**: Uses vector cross-product signed distances with state hysteresis (`last_crossed` direction caching). An object crossing once is registered, and cannot double-count while lingering near the boundary, while still allowing legitimate re-entry.

---

## 🛠️ Project Structure

```
EdgeAI/
├── config/
│   └── config.yaml             # Central configuration (zones, thresholds, paths)
├── models/
│   ├── yolov8n.pt              # Base PyTorch model
│   ├── yolov8n.onnx            # Exported ONNX model (FP32)
│   └── yolov8n_int8.onnx       # Quantized INT8 ONNX model
├── videos/
│   └── sample1.mp4             # High-resolution surveillance sample (2160x3840)
├── output/
│   ├── annotated/
│   │   ├── phase2_result.mp4   # Detection output
│   │   ├── phase3_result.mp4   # Multi-object tracking output
│   │   └── phase4_result.mp4   # Zone analytics, alerts, and counting output
│   ├── logs/
│   │   ├── surveillance.db     # SQLite event database
│   │   └── events.csv          # CSV event logs
│   └── benchmark_report.md     # Runtime performance comparison report
├── src/
│   ├── detect.py               # Phase 2: YOLO video detection
│   ├── tracker.py              # Phase 3: ByteTrack multi-object tracking
│   ├── zones.py                # Phase 4 & 5: Surveillance analytics & logger pipeline
│   ├── logger.py               # Phase 5: SQLite & CSV event storage
│   ├── benchmark.py            # Phase 6: Edge optimization benchmark
│   ├── calibrate_zones.py      # Visual zone selection & calibration tool
│   └── utils.py                # Geometry math, line tracking, HUD rendering
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Quick Start & Usage

### 1. Environment Setup
```bash
# Clone and enter directory
cd /Users/ashaf/EdgeAI

# Activate Python virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Object Detection (Phase 2)
```bash
python src/detect.py
```

### 3. Run Multi-Object ByteTrack Tracking (Phase 3)
```bash
python src/tracker.py
```

### 4. Run Smart City Surveillance Engine (Phase 4 & 5)
Executes crosswalk crowd monitoring, overcrowding alerting, roadway intrusion detection, virtual line counting, and writes events to SQLite and CSV:
```bash
python src/zones.py
```

### 5. Live Edge ONNX Surveillance Engine (Capstone Live Demo)
Launches the live GUI window running directly on the quantized **INT8 ONNX model** (3.34 MB) using ONNX Runtime:
```bash
# Live GUI surveillance window on CCTV video
python src/edge_infer.py

# Live GUI surveillance using your MacBook camera (Live Demo)
python src/edge_infer.py --source 0
```

### 6. Calibrate Custom Zones Visually
Launch the interactive calibration tool to click and save custom polygons directly to `config/config.yaml`:
```bash
# Visual interactive calibration window
python src/calibrate_zones.py

# Non-interactive configuration verification snapshot
python src/calibrate_zones.py --verify
```

### 7. Query Surveillance Event Database (Phase 5)
```bash
# View summary breakdown of surveillance events
python src/logger.py --summary

# View recent 20 logged events
python src/logger.py --recent 20
```

### 8. Run Edge Optimization Benchmark (Phase 6)
```bash
python src/benchmark.py --runs 50
```

---

## 🗄️ Database Schema (`surveillance.db`)

Events are automatically stored in the `events` table:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    frame_idx INTEGER,
    object_class TEXT,
    track_id INTEGER,
    event_type TEXT NOT NULL,   -- 'enter', 'exit', 'overcrowding', 'restricted_zone', 'crowd_cleared'
    zone TEXT,
    confidence REAL,
    x INTEGER,
    y INTEGER,
    details TEXT
);
```

Sample query:
```sql
SELECT event_type, object_class, track_id, timestamp, details 
FROM events 
WHERE event_type IN ('overcrowding', 'restricted_zone')
ORDER BY id DESC LIMIT 10;
```

---

## 📦 Edge Hardware Deployment Guidelines (Phase 7)

### NVIDIA Jetson (Nano / Orin Nano)
1. **Runtime**: TensorRT (`format='engine'`).
2. **Setup**:
   ```bash
   yolo export model=models/yolov8n.pt format=engine device=0 half=True
   ```
3. **Execution**: Run with CUDA execution provider for real-time 40–60+ FPS performance.

### Raspberry Pi 5
1. **Runtime**: ONNX Runtime with CPUExecutionProvider (or Hailo-8 AI accelerator HAT).
2. **Model**: Deploy `models/yolov8n_int8.onnx` to maximize throughput on ARM Cortex-A76.
3. **Input stream**: Use CSI camera / USB camera via `cv2.VideoCapture(0)`.

### Intel NUC / OpenVINO
1. **Export**:
   ```bash
   yolo export model=models/yolov8n.pt format=openvino half=True
   ```
2. **Acceleration**: Leverages Intel GPU / VPU via OpenVINO Execution Provider.

---

## ⚖️ Limitations & Future Improvements

1. **Perspective & Far-Field Pedestrians**: YOLOv8n is ultra-fast, but very small/distant pedestrians (>80m from camera) may experience intermittent misses. A specialized crowd-density estimation head or tiling inference (SAHI) can further improve far-field sensitivity.
2. **Camera Vibration / Panning**: Current polygon coordinates assume fixed CCTV perspectives. Dynamic perspective correction via optical flow or homography estimation can adapt to PTZ (Pan-Tilt-Zoom) cameras.
3. **Multi-Camera Edge Mesh**: Extending persistent IDs across multiple cameras using Edge Re-ID (Re-Identification) feature vectors.
