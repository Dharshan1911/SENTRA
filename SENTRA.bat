@echo off
echo Starting SENTRA...
cd /d "%~dp0"

IF NOT EXIST ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing/Verifying dependencies...
pip install -r requirements.txt

IF NOT EXIST "models\ppe_yolo11s\weights\best.pt" (
    echo Model not found. Starting training process...
    python training\train.py
)

echo Starting backend...
start "SENTRA Backend" cmd /c "call .venv\Scripts\activate.bat && uvicorn app.main:app --host 127.0.0.1 --port 8000"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Opening browser...
start http://127.0.0.1:8000

echo SENTRA is running!
