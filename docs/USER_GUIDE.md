# Operational User Guide

## AI-Powered Real-Time Traffic Monitoring and Vehicle Analytics System

This manual provides instructions for utilizing all features of the AI Traffic Monitoring dashboard.

---

### 1. Dashboard Navigation Overview

The application features a persistent left sidebar providing navigation between 6 primary views:

1. **📊 Overview**: High-level telemetry, total audited sessions, category breakdown, and recent processing history.
2. **📹 Live Monitoring**: Real-time inference feed for local USB/laptop webcams and authorized RTSP/HTTP IP camera streams.
3. **🎬 Video Analysis**: Upload pre-recorded traffic video or run synthetic demo feeds with frame-by-frame progress, annotated video playback, and CSV exports.
4. **📈 Traffic Analytics**: Granular session drill-down, temporal activity trend lines, and processing throughput metrics.
5. **📜 History**: Searchable and filterable database audit log with full CSV export capabilities.
6. **⚙️ Settings**: Detection thresholds, counting line position adjustments, hardware diagnostics, and database verification.

---

### 2. How to Analyze a Video File

1. Navigate to **🎬 Video Analysis** in the sidebar.
2. Under the **📁 Upload Traffic Video** tab, click **Browse files** and select an `.mp4`, `.avi`, `.mov`, or `.mkv` file.
   *(Alternatively, under the **🧪 Use Synthetic Sample Video** tab, click **Generate & Select Synthetic Sample Feed** if you wish to run a quick test without an external video file).*
3. Inspect the verified **Input Video Properties** (Resolution, FPS, Total Frames, Duration).
4. Configure analysis parameters:
   - **Encode & Save Annotated Output Video**: Enable to generate a downloadable MP4 video with bounding boxes and line overlays.
   - **Show Preview During Processing**: Enable live frame updates during inference.
   - **Process Every Nth Frame**: Set to 1 for full frame analysis, or 2-3 to speed up inference on low-power CPUs.
5. Click **🚀 Start Video Analysis**.
6. Track progress on the progress bar.
7. Upon completion:
   - Review **Final Session Analytics** (Total Vehicles, Inbound Flow, Outbound Flow, Pedestrians, Average FPS).
   - Click **📄 Download Session Metrics (CSV)** to download structured numerical data.
   - Click **🎬 Download Annotated Video** to download the rendered video file.

---

### 3. How to Use Live Monitoring

1. Navigate to **📹 Live Monitoring** in the sidebar.
2. Select your **Source Type**:
   - **Local Webcam**: Enter device index (`0` for built-in camera, `1` for external USB webcam).
   - **IP / CCTV Stream**: Enter an authorized RTSP stream URL (`rtsp://user:pass@192.168.1.100:554/ch0`).
3. Optionally check **Save Live Video** if you wish to record the live monitoring session to disk.
4. Click **▶️ Start Monitoring**.
5. The live feed will display:
   - Detected vehicles and pedestrians with color-coded bounding boxes.
   - Persistent ByteTrack IDs (e.g. `#1`, `#2`).
   - Trajectory motion trails displaying path history.
   - Cyan virtual counting line and semi-transparent count HUD in top-left.
   - Live FPS, Inbound/Outbound counts, and active congestion badge (`LOW`, `MODERATE`, `HIGH`).
6. When finished, click **⏹️ Stop Monitoring**. The session will be automatically finalized and stored in SQLite.

---

### 4. How Vehicle Counting & Line Crossing Works

- The system establishes a **Virtual Counting Line** across the video frame (configurable via the sidebar slider or settings).
- Rather than counting an object merely because it enters the screen, the system tracks each object's 2D trajectory.
- When an object's bottom-center reference point intersects the counting line segment:
  - Downward motion across the line increments the **Inbound** counter.
  - Upward motion across the line increments the **Outbound** counter.
  - The object's unique ID is registered in the session's crossed set, **strictly preventing duplicate counting** on subsequent frames.

---

### 5. Exporting Reports and Audit Records

1. Navigate to **📜 History** in the sidebar.
2. Use the search input or date filter to locate sessions of interest.
3. Click **📥 Export All Records (CSV)** to generate a comprehensive spreadsheet of all past traffic audits.
4. Each exported CSV contains:
   - `session_id`
   - `source_type` & `source_name`
   - `start_time` & `end_time`
   - `duration_sec`
   - `total_vehicles`, `cars`, `buses`, `trucks`, `motorcycles`, `pedestrians`
   - `inbound_count` & `outbound_count`
   - `avg_fps`
