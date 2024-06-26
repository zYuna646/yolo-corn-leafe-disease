import os
import shutil
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))

# Path asal data
data_path = os.path.abspath(os.path.join(current_dir, '../../data/raw'))
categories = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
train_ratio = 0.8

# Path tujuan
dataset_path = os.path.abspath(os.path.join(current_dir, '../../data/processed'))
train_img_dir = os.path.join(dataset_path, 'images/train')
val_img_dir = os.path.join(dataset_path, 'images/val')
train_lbl_dir = os.path.join(dataset_path, 'labels/train')
val_lbl_dir = os.path.join(dataset_path, 'labels/val')

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

for category in categories:
    category_path = os.path.join(data_path, category)
    images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]
    labels = [img.replace('.jpg', '.txt') for img in images]

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, train_size=train_ratio)

    for img, lbl in zip(train_images, train_labels):
        shutil.copy(img, train_img_dir)
        shutil.copy(lbl, train_lbl_dir)

    for img, lbl in zip(val_images, val_labels):
        shutil.copy(img, val_img_dir)
        shutil.copy(lbl, val_lbl_dir)
