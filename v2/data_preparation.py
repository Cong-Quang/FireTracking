import os
import shutil
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(base_dir, output_dir, val_size=0.15, test_size=0.15):
    """
    Explore the dataset, perform EDA, and split into train/val/test sets.
    """
    classes = ['fire_images', 'non_fire_images']
    data = []
    labels = []
    
    # --- EDA: Count and distribution ---
    print("--- Exploratory Data Analysis ---")
    counts = {}
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Directory not found: {cls_dir}")
            continue
            
        images = glob.glob(os.path.join(cls_dir, '*.[pj]*g')) # jpg, png, jpeg
        counts[cls] = len(images)
        print(f"Class '{cls}' has {len(images)} images.")
        
        for img_path in images:
            data.append(img_path)
            labels.append(cls)

    if not data:
        print("No images found. Exiting.")
        return

    # Plot Distribution
    plt.figure(figsize=(8, 6))
    plt.bar(counts.keys(), counts.values(), color=['red', 'green'])
    plt.title('Dataset Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/class_distribution.png')
    print("Saved class distribution plot to results/class_distribution.png")

    # Visualize sample images
    plt.figure(figsize=(12, 6))
    for i, cls in enumerate(classes):
        cls_images = [img for img, label in zip(data, labels) if label == cls]
        if cls_images:
            sample_img_path = cls_images[0]
            img = cv2.imread(sample_img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 2, i + 1)
                plt.imshow(img)
                plt.title(f"Sample: {cls}")
                plt.axis('off')
    
    plt.savefig('results/sample_images.png')
    print("Saved sample images plot to results/sample_images.png")
    
    # --- Split Data ---
    # Split train and temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=(val_size + test_size), stratify=labels, random_state=42
    )
    
    # Split val and test
    val_ratio_in_temp = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_in_temp), stratify=y_temp, random_state=42
    )
    
    print(f"\n--- Data Split Info ---")
    print(f"Train Set: {len(X_train)} images")
    print(f"Validation Set: {len(X_val)} images")
    print(f"Test Set: {len(X_test)} images")

    # --- Copy to Output Directory ---
    print("\nCopying files to dataset directory...")
    splits = [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]
    
    for split_name, X_split, y_split in splits:
        for img_path, label in zip(X_split, y_split):
            # new label name: 'fire' or 'non_fire'
            new_label = 'fire' if 'non' not in label else 'non_fire'
            dest_dir = os.path.join(output_dir, split_name, new_label)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Use original filename
            filename = os.path.basename(img_path)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(img_path, dest_path)
            
    print(f"Data preparation complete. Output saved to {output_dir}/")

if __name__ == "__main__":
    BASE_DIR = "." # Current directory contains fire_images and non_fire_images
    OUTPUT_DIR = "dataset"
    prepare_data(BASE_DIR, OUTPUT_DIR)
