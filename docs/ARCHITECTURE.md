# System Architecture Documentation

## AI-Powered Real-Time Traffic Monitoring and Vehicle Analytics System

### 1. High-Level System Architecture

The AI Traffic Monitoring System is built upon a decoupled, modular pipeline separating frame ingestion, deep learning inference, object tracking, spatial geometric counting, analytical aggregation, and user presentation.

```mermaid
flowchart LR
    subgraph Ingestion["Input Ingestion Layer"]
        V[Video File]
        W[Webcam Feed]
        IP[IP/CCTV RTSP Stream]
    end

    subgraph CoreEngine["Core CV & Analytics Engine"]
        VP[VideoProcessor\nOpenCV Frame Capture]
        YD[YOLODetector\nYOLOv8n PyTorch Model]
        BT[ByteTracker\nLinear Assignment & Kalman State]
        VC[VehicleCounter\n2D Geometric Line Intersection]
        TA[TrafficAnalyzer\nDensity & Congestion Heuristics]
    end

    subgraph Storage["Persistence Layer"]
        DB[(SQLite3 Database)]
        DISK[Processed Video & CSV Reports]
    end

    subgraph Presentation["Presentation Layer (Streamlit)"]
        OV[Overview Dashboard]
        LM[Live Monitoring Panel]
        VA[Video Analysis Interface]
        AN[Traffic Analytics View]
        HI[History & Audit Explorer]
        ST[System Diagnostics & Settings]
    end

    Ingestion --> VP
    VP --> YD
    YD --> BT
    BT --> VC
    VC --> TA
    TA --> DB
    TA --> DISK
    TA --> Presentation
    DB --> Presentation
```

---

### 2. Module Responsibilities

#### 2.1 Configuration (`config/settings.py`)
- Centralizes application settings through `AppConfig` dataclass.
- Manages directory paths (`models/`, `data/uploads/`, `data/processed/`, `data/samples/`, `logs/`, `reports/`).
- Defines target COCO class indices:
  - `0`: Person
  - `2`: Car
  - `3`: Motorcycle
  - `5`: Bus
  - `7`: Truck
- Defines visualization colors (BGR for OpenCV, Hex for Streamlit/Matplotlib).

#### 2.2 YOLO Object Detector (`core/detector.py`)
- Manages YOLOv8 model loading with automatic device detection (`cuda` if compatible PyTorch GPU is found, with reliable fallback to `cpu`).
- Pre-filters non-traffic classes at the inference layer using `classes=[0, 2, 3, 5, 7]` to conserve compute.
- Returns structured `Detection` objects containing bounding box, class, confidence, and bottom-center reference point.
- Draws high-contrast bounding boxes with class badges and background rectangles for maximum visibility.

#### 2.3 Object Tracker (`core/tracker.py`)
- Employs ByteTrack via Ultralytics' tracking API (`model.track(..., persist=True, tracker="bytetrack.yaml")`).
- Associates detections with persistent integer track IDs across video frames using Hungarian algorithm linear assignment (`lap`).
- Maintains trajectory centroid histories (`track_history`) over sliding windows for motion trail visualization and crossing calculations.
- Preserves track class memory to stabilize object classification across momentary occlusions.

#### 2.4 Vehicle Counter (`core/vehicle_counter.py`)
- Computes bidirectional virtual line crossings.
- Employs **exact 2D segment intersection**:
  Two segments $AB$ (counting line) and $CD$ (vehicle trajectory from frame $t-1$ to $t$) intersect if and only if points $C$ and $D$ lie on opposite sides of line $AB$, and endpoints $A$ and $B$ lie on opposite sides of trajectory line $CD$.
- Distinguishes **Inbound** (downward vertical movement) and **Outbound** (upward vertical movement).
- Avoids duplicate counting by maintaining a set of already-crossed track IDs (`crossed_track_ids`).
- Aggregates unique objects observed by class (`car`, `bus`, `truck`, `motorcycle`, `person`).

#### 2.5 Traffic Analyzer (`core/traffic_analyzer.py`)
- Computes vehicle surface density:
  $$\text{Density} = \min\left(1.0, \frac{\sum \text{Area}(\text{Vehicle Bounding Boxes})}{\text{Frame Width} \times \text{Frame Height}}\right)$$
- Categorizes congestion using configurable heuristics:
  - **LOW**: Active vehicles $< 5$ and Density $< 0.15$
  - **MODERATE**: Active vehicles $5 \le N \le 15$ or Density $> 0.15$
  - **HIGH**: Active vehicles $> 15$ or Density $> 0.35$
- Maintains sliding window moving average for frames per second (FPS) throughput.
- Generates periodic analytical snapshots saved to the database.

#### 2.6 Persistence Manager (`database/db_manager.py`)
- Embedded SQLite engine utilizing standard parameterized SQL queries.
- Manages `sessions` table (metadata, duration, class totals, inbound/outbound counts, average FPS).
- Manages `traffic_snapshots` table (temporal snapshots over frame sequence).
- Foreign key cascading deletion prevents orphaned snapshot records.

---

### 3. Data Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit UI
    participant VP as VideoProcessor
    participant YD as YOLODetector
    participant BT as ByteTracker
    participant VC as VehicleCounter
    participant TA as TrafficAnalyzer
    participant DB as SQLite Database

    User->>UI: Select Video / Start Monitoring
    UI->>DB: create_session(session_id, source_type, source_name)
    UI->>VP: open() & frame_generator()
    loop For Every Frame
        VP->>UI: yield (frame_idx, frame)
        UI->>BT: track(frame)
        BT->>YD: detect(frame)
        YD-->>BT: Detections
        BT-->>UI: Detections with Track IDs
        UI->>VC: process_frame(detections, frame_shape)
        VC-->>UI: CountingState (Inbound, Outbound, Classes)
        UI->>TA: record_frame(frame_idx, detections, counting_state, elapsed)
        TA-->>UI: TrafficSnapshot
        alt Every 20-30 Frames
            UI->>DB: save_snapshot(session_id, snapshot_data)
        end
        UI->>BT: draw_tracks(frame, detections)
        UI->>VC: draw_counting_overlay(annotated_frame)
        UI->>User: Render Frame & Live KPIs
    end
    UI->>VP: close()
    UI->>DB: finalize_session(session_id, final_metrics)
    UI->>User: Display Final Summary & Download Buttons (CSV / Video)
```

---

### 4. Security & Privacy Architecture

1. **Credential Sanitization**:
   Stream URLs containing basic authentication credentials (e.g., `rtsp://admin:password@host:port/stream`) are sanitized using regular expressions (`utils/logger.py:sanitize_url`) before writing to logs, displaying in the UI, or saving to the SQLite database.
2. **File Upload Isolation**:
   Uploaded files are sanitized using `sanitize_filename` which strips directory traversal tokens (`../`, `..\`) and replaces illegal punctuation with underscores. Upload sizes are constrained by `max_upload_size_mb`.
3. **No Dynamic Execution**:
   Uploaded files are treated strictly as read-only binary streams via OpenCV's `VideoCapture` decoder; no executable or script execution is ever permitted.
