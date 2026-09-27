# SENTRA (Safety Equipment & Noncompliance Tracking using Real-time AI)

SENTRA is a robust, real-time Computer Vision application that detects Personal Protective Equipment (PPE) compliance on construction sites. Built with **YOLO11** and **FastAPI**, it allows for Image, Video, and live Webcam streaming to monitor safety compliance.

## Features
- **Real-Time Detection**: Detects Helmets, Vests, Gloves, Boots, and Goggles.
- **Safety Logic Dashboard**: Calculates overall compliance and highlights safety violations immediately.
- **Multiple Inputs**: Drag-and-drop Image processing, incremental Video processing, and ultra-low latency WebSocket Webcam streaming.
- **1-Click Launch**: Fully automated setup and launch using Windows Batch scripts.

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dharshan1911/SENTRA.git
   cd SENTRA
   ```

2. **One-Click Start**:
   Simply double-click the **`SENTRA.bat`** file in the root folder.
   
   *What this does:*
   - Creates a Python virtual environment (`.venv`).
   - Installs all required dependencies from `requirements.txt`.
   - Starts the FastAPI backend.
   - Automatically opens your web browser to the dashboard (`http://127.0.0.1:8000`).

## Retraining the Model (Optional)
This repository includes a pre-trained YOLO11s model (`models/ppe_yolo11s/weights/best.pt`) optimized for 15 epochs at 320px resolution. If you wish to retrain the model on your own hardware for higher accuracy:

1. Download the dataset from [Kaggle: PPE Kit Detection](https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers).
2. Extract the `train`, `val`, and `test` folders into `SENTRA/datasets/images/` and `SENTRA/datasets/labels/` appropriately.
3. Open a terminal in the SENTRA folder and run:
   ```cmd
   call .venv\Scripts\activate.bat
   python training\train.py
   ```
4. The new `best.pt` will automatically overwrite the old one, and `SENTRA.bat` will use your new model on the next launch.

## Built With
- **Ultralytics YOLO11** (Computer Vision)
- **FastAPI & Uvicorn** (Backend API & WebSocket server)
- **OpenCV** (Image and Frame processing)
- **HTML5 / CSS / Vanilla JS** (Frontend)
