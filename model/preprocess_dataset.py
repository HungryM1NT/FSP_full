import os
import json
import cv2
import shutil
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = "./SROIE2019"

OUTPUT_DIR = "./processed_sroie"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

SEED = 54321

def create_dirs():
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(os.path.join(split_dir, "img"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "entities"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "box"), exist_ok=True)

def preprocess_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # processed_img = cv2.adaptiveThreshold(
    #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    cv2.imwrite(save_path, gray)

def load_and_group_data():
    data = []
    
    for split_folder in ["train", "test"]:
        entities_dir = os.path.join(DATA_DIR, split_folder, "entities")
        
        if not os.path.exists(entities_dir):
            continue
            
        for filename in os.listdir(entities_dir):
            if not filename.endswith(".txt"):
                continue
                
            file_id = filename.replace(".txt", "")
            txt_path = os.path.join(entities_dir, filename)
            img_path = os.path.join(DATA_DIR, split_folder, "img", f"{file_id}.jpg")
            box_path = os.path.join(DATA_DIR, split_folder, "box", f"{file_id}.txt")
            
            if not os.path.exists(img_path):
                continue
                
            with open(txt_path, 'r', encoding='utf-8') as f:
                try:
                    content = json.load(f)
                    company_name = content.get("company", "UNKNOWN_COMPANY")
                except json.JSONDecodeError:
                    continue
                    
            data.append({
                "file_id": file_id,
                "img_path": img_path,
                "txt_path": txt_path,
                "box_path": box_path,
                "company": company_name
            })
            
    return pd.DataFrame(data)

def split_and_process_data(df):
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
    train_idx, val_test_idx = next(gss_train.split(df, groups=df['company']))
    
    train_df = df.iloc[train_idx]
    val_test_df = df.iloc[val_test_idx]
    
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED)
    val_idx, test_idx = next(gss_val.split(val_test_df, groups=val_test_df['company']))
    
    val_df = val_test_df.iloc[val_idx]
    test_df = val_test_df.iloc[test_idx]
    
    splits = {
        "train": (train_df, TRAIN_DIR),
        "val": (val_df, VAL_DIR),
        "test": (test_df, TEST_DIR)
    }
    
    for (split_df, target_dir) in splits.values():
        for _, row in split_df.iterrows():
            new_img_path = os.path.join(target_dir, "img", f"{row['file_id']}.png")
            new_txt_path = os.path.join(target_dir, "entities", f"{row['file_id']}.txt")
            new_box_path = os.path.join(target_dir, "box", f"{row['file_id']}.txt")
            
            preprocess_image(row['img_path'], new_img_path)
            shutil.copy2(row['txt_path'], new_txt_path)
            
            if os.path.exists(row['box_path']):
                shutil.copy2(row['box_path'], new_box_path)

if __name__ == "__main__":
    print("Загрузка датасета")
    create_dirs()
    df = load_and_group_data()
    print(f"{len(df)} документов. Уникальных гурпп: {df['company'].nunique()}")
    split_and_process_data(df)