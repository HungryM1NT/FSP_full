import os
import cv2
from tqdm import tqdm

IMG_DIR = "processed_sroie/val/img"
BOX_DIR = "processed_sroie/val/box"

OUTPUT_DATASET_DIR = "easyocr_dataset/val"
CROP_DIR = os.path.join(OUTPUT_DATASET_DIR, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

labels = []

files = [f for f in os.listdir(BOX_DIR) if f.endswith(".txt")]

for filename in tqdm(files):
    file_id = filename.replace(".txt", "")
    img_path = os.path.join(IMG_DIR, f"{file_id}.png")
    box_path = os.path.join(BOX_DIR, filename)
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    try:
        with open(box_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(box_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
    for idx, line in enumerate(lines):
        parts = line.strip().split(',', 8)
        if len(parts) < 9:
            continue
            
        try:
            x_coords = [int(parts[0]), int(parts[2]), int(parts[4]), int(parts[6])]
            y_coords = [int(parts[1]), int(parts[3]), int(parts[5]), int(parts[7])]
            
            x_min, x_max = max(0, min(x_coords)), min(img.shape[1], max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(img.shape[0], max(y_coords))
            
            text = parts[8].strip()
            
            crop_img = img[y_min:y_max, x_min:x_max]
            
            if crop_img.size == 0 or not text:
                continue
                
            crop_filename = f"{file_id}_{idx}.jpg"
            crop_filepath = os.path.join(CROP_DIR, crop_filename)
            
            cv2.imwrite(crop_filepath, crop_img)
            labels.append(f"{crop_filename}\t{text}")
            
        except Exception as e:
            continue

with open(os.path.join(OUTPUT_DATASET_DIR, "labels.txt"), 'w', encoding='utf-8') as f:
    f.write("\n".join(labels))
