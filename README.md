# 🛰️ Satellite Cloud Removal & Multi-Sector Analysis System
**Smart India Hackathon (SIH 2026) | Problem Statement ID: 26209**  
*Theme: Space Technology | Category: Software*

---

## 🌟 Executive Summary & Problem Addressed
Over 60–70% of Earth is obscured by clouds at any given moment, rendering critical satellite remote sensing imagery unusable for agriculture, disaster relief, infrastructure mapping, and climate analysis.

This system provides an **end-to-end, high-performance, open-source web application** that transforms cloud-obscured satellite imagery into crystal-clear optical data using Computer Vision and physical optical dehazing models, while immediately deriving multi-sector actionable intelligence.

---

## 🚀 Key Features

1. **Precision Optical Dehazing & Enhancement Pipeline**:
   - **Dark Channel Prior (DCP)**: Accurate atmospheric haze transmission map estimation.
   - **Guided Filtering**: Edge-preserving refinement of optical transmission.
   - **LAB Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Local contrast restoration.
   - **Multi-Scale Unsharp Masking**: High-frequency terrain detail amplification.
   - **Cloud Mask Generation**: Multi-spectral morphological segmentation of cloud perimeters.

2. **Real-Time Telemetry & Image Metrics**:
   - Processing time, input/output file sizes, and contrast improvement percentage.
   - Haze attenuation index and estimated atmospheric cloud coverage percentage.
   - **Dual Pixel Intensity Histogram**: Dynamic before (cloudy) vs after (cleared) brightness curves.

3. **Interactive Optical Comparison Viewer**:
   - Draggable before/after split slider with smooth cursor tracking.
   - Toggleable cloud mask overlay.
   - View modes: Split View, Original Cloudy, and Cleared Optical.

4. **Multi-Sector Post-Processing CV Suite**:
   - 🌿 **NDVI Vegetation Index (Agriculture)**: Healthy, moderate, and barren vegetation partitioning, mean NDVI score, and donut chart.
   - 🌊 **Flood & Water Inundation (Disaster Management)**: Surface water detection, flood threat classification, perimeter risk contours.
   - 🏠 **Building Extraction (Urban Planning)**: Structural edge analysis, building footprint counts, and urban density rating.
   - 🗺️ **Land Cover Classification (Climate Research)**: 4-class biophysical segmentation (Vegetation, Water, Bare Soil, Urban) and carbon sink rating.

5. **🌐 Global Region Fingerprint (Radar / Spider Chart)**:
   - Unified 4-axis polar radar chart tracking Visibility Score, Vegetation Index, Hydrological Index, and Urban Footprint.

6. **Executive Mission Telemetry Dashboard**:
   - Aggregated mission statistics, surface area analyzed, cumulative flood alerts, and pipeline architecture diagram.

7. **Export & Delivery**:
   - Download cleared high-resolution images.
   - Instant Executive Mission PDF Report generator with embedded before/after and sector breakdowns.

---

## 📁 Project Architecture & File Organization

```
satellite-cloud-removal/
│
├── app.py                         # Flask REST API server (routes, file uploads, sample presets, exports)
├── cloud_removal.py               # Optical dehazing & enhancement engine (DCP, CLAHE, Metrics, Histogram)
├── analysis_engine.py             # 4 Computer Vision modules (NDVI, Flood, Buildings, Land Cover)
├── sample_generator.py            # Generates realistic synthetic satellite scenes (Agriculture, Coastal, Urban, Mountain)
├── report_generator.py            # Executive PDF report generator with embedded side-by-side images
├── test_system.py                 # Automated end-to-end verification script
├── requirements.txt               # Python package dependencies
├── setup.bat                      # One-click Windows installation batch file
├── run.bat                        # One-click Windows server launch batch file
│
├── templates/
│   └── index.html                 # Main dashboard UI (UTF-8 compliant, responsive, Chart.js, Lucide icons)
│
└── static/
    ├── css/
    │   └── style.css              # Aerospace deep cosmos dark theme, glassmorphism, glowing telemetry
    ├── js/
    │   └── app.js                 # Split slider dragging, Chart.js graphs, radar fingerprint, AJAX API calls
    ├── samples/                   # Pre-generated sample satellite scenes (Agri, Coastal, Urban, Mountain)
    ├── uploads/                   # Staged user-uploaded satellite images
    ├── results/                   # Enhanced clear images and cloud masks
    ├── analysis/                  # Colorized post-processing analysis maps with HUD legends
    └── reports/                   # Generated executive PDF mission reports
```

---

## 🛠️ Installation & Quick Start

### Prerequisites
- Python 3.10 or higher
- Windows, Linux, or macOS

### One-Click Launch (Windows)
1. Double-click `setup.bat` (or run `setup.bat` in CMD) to install requirements and generate sample scenes.
2. Double-click `run.bat` to start the local Flask server and automatically launch `http://127.0.0.1:5000` in your browser.

### Manual Command-Line Launch
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample scenes (optional if already generated)
python sample_generator.py

# 3. Launch Flask server
python app.py
```
Open **`http://127.0.0.1:5000`** in any web browser.

---

## 🔬 Testing & Verification
Run the automated verification suite:
```bash
python test_system.py
```
Expected output:
```
==================================================
  RUNNING SATELLITE SYSTEM PIPELINE VERIFICATION  
==================================================
[Step 1] Ensuring sample scenes exist...
  -> Sample scenes verified.
[Step 2] Testing CloudRemovalEngine enhancement...
  -> Enhancement successful! Time: 0.232s, Contrast Boost: +68.1%, Cloud Cover: 34.9%
[Step 3] Testing 4 Sector CV Analysis modules...
  -> NDVI OK! Dense Veg: 24.5%, Mean NDVI: 0.15
  -> Flood OK! Water Coverage: 4.1%, Bodies: 3
  -> Buildings OK! Detected: 0, Built-up Density: 0.0%
  -> Land Cover OK! Forest/Veg: 61.4%, Urban: 22.7%
[Step 4] Testing Executive PDF Report generation...
  -> PDF Report OK! Size: 1453.9 KB
==================================================
  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!    
==================================================
```
