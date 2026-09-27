import os
import cv2
import numpy as np
from ultralytics import YOLO

class PPEDetector:
    def __init__(self, model_path="models/ppe_yolo11s/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        
        # Load model if it exists
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"WARNING: Model not found at {self.model_path}. Please train the model first.")

        # Classes mapping from Kaggle notebook
        self.classes = {
            0: "Helmet",   1: "Gloves",    2: "Vest",
            3: "Boots",    4: "Goggles",   5: "none",
            6: "Person",   7: "no_helmet", 8: "no_goggle",
            9: "no_gloves", 10: "no_boots",
        }
        
        # Positive and negative classes
        self.positive_classes = {"Helmet", "Gloves", "Vest", "Boots", "Goggles", "Person"}
        self.negative_classes = {"no_helmet", "no_goggle", "no_gloves", "no_boots"}

        # Colors for drawing
        self.colors = {
            "Helmet": (0, 255, 0),
            "Gloves": (0, 255, 0),
            "Vest": (0, 255, 0),
            "Boots": (0, 255, 0),
            "Goggles": (0, 255, 0),
            "Person": (255, 255, 0),
            "no_helmet": (0, 0, 255),
            "no_goggle": (0, 0, 255),
            "no_gloves": (0, 0, 255),
            "no_boots": (0, 0, 255),
            "none": (128, 128, 128)
        }

    def predict(self, image):
        if self.model is None:
            # Return dummy if model is missing
            return [], image, 0, 0, False

        # Run inference
        results = self.model(image, conf=0.25, iou=0.45)[0]
        
        detections = []
        persons = 0
        violations = 0
        
        annotated_img = image.copy()
        
        # Parse results
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.classes.get(cls_id, "unknown")
            
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })
            
            if cls_name == "Person":
                persons += 1
            if cls_name in self.negative_classes:
                violations += 1

            # Draw bounding box
            color = self.colors.get(cls_name, (255, 255, 255))
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf*100:.1f}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        compliant = persons - violations if persons > violations else 0
        is_safe = (violations == 0 and persons > 0)
        
        return detections, annotated_img, persons, violations, is_safe
