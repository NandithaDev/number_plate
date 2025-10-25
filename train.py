from ultralytics import YOLO

def run_training():
    model = YOLO("yolov8n.pt")  
    model.train(
        data="data.yaml",
        epochs=70,
        imgsz=640,
        batch=8,
        workers=2,  
        device=0
    )
    

if __name__ == "__main__":  
    run_training()
