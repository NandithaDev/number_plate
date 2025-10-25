import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm

# === 1. Load model ===
model = YOLO("C:\\Users\\Nanditha D\\Desktop\\projects\\number plate detection\\runs\\detect\\train\\weights\\best.pt")
# === 2. Setup paths ===
input_folder = "input_frames"
output_folder = "output_plates"
os.makedirs(output_folder, exist_ok=True)

# === 3. Get all image files ===
valid_exts = (".jpg", ".jpeg", ".png")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

# === 4. Loop through frames ===
for file in tqdm(image_files, desc="Processing frames"):
    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # === 5. Run prediction ===
    results = model.predict(img, conf=0.4, verbose=False)

    # === 6. Process detections ===
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
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            save_name = f"{os.path.splitext(file)[0]}_{i}_cls{cls}_conf{conf:.2f}.jpg"
            cv2.imwrite(os.path.join(output_folder, save_name), crop)

print(f"\n✅ Cropped license plates saved to '{output_folder}'")
