import re
import yt_dlp
import numpy as np
import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import pytesseract

video_path=r"C:\Users\Nanditha D\Desktop\projects\number plate detection\vid1.mp4"


def extract_motion_frames(video_path, output_dir="motion_frames", diff_threshold=20000):
    model = YOLO(r"C:\Users\Nanditha D\Desktop\projects\number plate detection\runs\detect\train\weights\best.pt")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: cannot read video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    saved_count = 0
    frame_count = 0
    detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        diff_sum = diff.sum()

        if diff_sum > diff_threshold:
            results = model.predict(frame, conf=0.4, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0:
                        continue
                    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                    extracted_text = pytesseract.image_to_string(crop)
                    cleaned_text = re.sub(r'\s+', '', extracted_text)
                    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', cleaned_text)

                    if cleaned_text:
                        print(f"[{frame_count}] Detected Plate: {cleaned_text}")
                        detected_count += 1

        prev_gray = gray
        frame_count += 1

    cap.release()
    print(f"Total motion frames processed: {frame_count}")
    print(f"Total detected plates: {detected_count}")


# Example usage
extract_motion_frames("vid1.mp4", diff_threshold=20000)
