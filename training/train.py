import os
from ultralytics import YOLO
import yaml

# CFG from Kaggle Notebook
PPE_CLASSES = {
    0: "Helmet",   1: "Gloves",    2: "Vest",
    3: "Boots",    4: "Goggles",   5: "none",
    6: "Person",   7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots",
}

def create_yaml(dataset_path):
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    
    # Assume the standard structure if it exists
    data = {
        "train": os.path.abspath(os.path.join(dataset_path, "images", "train")),
        "val": os.path.abspath(os.path.join(dataset_path, "images", "val")),
        "test": os.path.abspath(os.path.join(dataset_path, "images", "test")),
        "nc": len(PPE_CLASSES),
        "names": list(PPE_CLASSES.values())
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
        
    return yaml_path

def train_model():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    
    # Check if dataset exists
    train_dir = os.path.join(dataset_path, "images", "train")
    if not os.path.exists(train_dir):
        print(f"Dataset not found at {train_dir}")
        print("Please download the Kaggle dataset 'ppe-kit-detection-construction-site-workers' and extract it into the 'datasets' folder.")
        return

    yaml_path = create_yaml(dataset_path)
    print(f"Created yaml configuration at {yaml_path}")

    # Initialize model
    model = YOLO("yolo11s.pt")

    # Train model
    print("Starting training with Kaggle notebook parameters...")
    results = model.train(
        data=yaml_path,
        epochs=15,
        batch=16,
        imgsz=320,
        optimizer="AdamW",
        lr0=0.01,
        cos_lr=True,
        patience=5,
        project=os.path.abspath(os.path.join(dataset_path, "..", "models")),
        name="ppe_yolo11s",
        exist_ok=True
    )
    
    print("Training complete. Model saved in models/ppe_yolo11s/weights/best.pt")

if __name__ == "__main__":
    train_model()
