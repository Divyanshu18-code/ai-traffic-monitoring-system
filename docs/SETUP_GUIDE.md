# Setup and Deployment Guide

## AI-Powered Real-Time Traffic Monitoring and Vehicle Analytics System

This guide outlines step-by-step instructions to set up, configure, and run the traffic monitoring application on a Windows development system.

---

### 1. Prerequisites

- **Operating System**: Windows 10 / 11 (64-bit)
- **Python**: Python 3.11 or Python 3.12 (Recommended: Python 3.12 64-bit)
- **Git**: Installed for version management (optional but recommended)
- **Hardware Requirements**:
  - Minimum: Intel Core i5 / AMD Ryzen 5, 8 GB RAM (CPU mode)
  - Recommended: Intel Core i7 / AMD Ryzen 7, 16 GB RAM, NVIDIA RTX GPU with CUDA 12+

---

### 2. Environment Setup (Windows PowerShell)

#### Step 1: Open PowerShell and navigate to the project directory
```powershell
cd d:\ai-traffic-monitoring-system
```

#### Step 2: Create a virtual environment
```powershell
py -3.12 -m venv venv
```

#### Step 3: Set script execution policy (if restricted)
If PowerShell prevents running scripts, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

#### Step 4: Activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(Your command prompt should now show `(venv)`)*

#### Step 5: Upgrade pip and install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Model Weights Setup

The application automatically downloads official Ultralytics YOLOv8 weights on the first run.

If you are working offline or in a restricted network environment:
1. Download `yolov8n.pt` manually from the official release:
   - [yolov8n.pt Download Link](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt)
2. Place the file into the `models/` directory:
   ```powershell
   Copy-Item -Path path\to\downloaded\yolov8n.pt -Destination models\yolov8n.pt
   ```

---

### 4. Running the Application

Launch the Streamlit dashboard:
```powershell
streamlit run app.py
```

Once running, access the user interface in your web browser:
```
Local URL: http://localhost:8501
```

---

### 5. Running Automated Unit Tests

To execute the test suite and verify all counting, analytics, database, and validation logic:
```powershell
python -m pytest -q
```

Expected output:
```
...........................                                              [100%]
27 passed in ~4s
```

---

### 6. Common Troubleshooting

#### Issue 1: `AttributeError: module 'cv2' has no attribute 'VideoWriter_fourcc'`
- **Cause**: Incomplete or broken OpenCV installation.
- **Solution**: Reinstall `opencv-python` inside the virtual environment:
  ```powershell
  pip install --force-reinstall opencv-python
  ```

#### Issue 2: `PermissionError: [WinError 32]` during video export
- **Cause**: Another media player or process has locked the video file.
- **Solution**: Ensure previous video players are closed before re-processing. The application generates unique file names for each session to avoid collision.

#### Issue 3: Camera access denied
- **Cause**: Windows Privacy Settings blocking camera access to Python.
- **Solution**: Open Windows **Settings -> Privacy & Security -> Camera** and toggle **"Let desktop apps access your camera"** to **On**.
