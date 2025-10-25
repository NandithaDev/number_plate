import cv2
import pytesseract
import os
from os import listdir
import re

# get the path/directory
folder_dir = "C:/Users/Nanditha D/Desktop/projects/number plate detection/output_plates"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
for images in os.listdir(folder_dir):

    image_path = os.path.join(folder_dir, images)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    extracted_text = pytesseract.image_to_string(image_rgb)
    cleaned_text = re.sub(r'\s+', '', extracted_text)  # removes all spaces, tabs, newlines
    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', cleaned_text)  # keeps only letters and numbers

    
    print(cleaned_text)