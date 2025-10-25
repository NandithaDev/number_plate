import cv2
import os
from matplotlib import image
from ultralytics import YOLO
from tqdm import tqdm
import pytesseract
from os import listdir
import re

folder_dir = "C:/Users/Nanditha D/Desktop/projects/number plate detection/output_plates"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_frames(video_path, output_dir="input_frames", interval_seconds=3):
    
    model = YOLO("C:\\Users\\Nanditha D\\Desktop\\projects\\number plate detection\\runs\\detect\\train\\weights\\best.pt")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps * interval_seconds)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # === 5. Run prediction ===
            results = model.predict(frame, conf=0.4, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

        # move once to CPU and get numpy arrays
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    cls = clss[i]
                    conf = confs[i]

                    # crop and save
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0:
                        continue

                    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)


                    extracted_text = pytesseract.image_to_string(image_rgb)
                    cleaned_text = re.sub(r'\s+', '', extracted_text)  # removes all spaces, tabs, newlines
                    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', cleaned_text)  # keeps only letters and numbers

                    print(cleaned_text)

        

    cap.release()
    print(f"Total frames saved: {saved_count}")

# Example usage
extract_frames("vid1.mp4", interval_seconds=1)

