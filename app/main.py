import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import base64
import time
from app.detector import PPEDetector

app = FastAPI(title="SENTRA - Safety Equipment & Noncompliance Tracking")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Get path to model relative to project root
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "ppe_yolo11s", "weights", "best.pt"))
detector = PPEDetector(model_path=model_path)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        return f.read()

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": detector.model is not None}

@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if detector.model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded. Please train first."})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    start_time = time.time()
    detections, annotated_img, persons, violations, is_safe = detector.predict(img)
    fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0

    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "detections": detections,
        "persons": persons,
        "violations": violations,
        "is_safe": is_safe,
        "fps": round(fps, 1),
        "image": f"data:image/jpeg;base64,{img_base64}"
    }

@app.websocket("/ws/webcam")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if detector.model is None:
        await websocket.send_json({"error": "Model not loaded."})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Expecting base64 image data
                if "," in data:
                    data = data.split(",", 1)[1]
                
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue

                start_time = time.time()
                detections, annotated_img, persons, violations, is_safe = detector.predict(img)
                fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                
                _, buffer = cv2.imencode('.jpg', annotated_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    "detections": detections,
                    "persons": persons,
                    "violations": violations,
                    "is_safe": is_safe,
                    "fps": round(fps, 1),
                    "image": f"data:image/jpeg;base64,{img_base64}"
                })
            except Exception as frame_e:
                print(f"Frame processing error: {frame_e}")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket: {e}")
