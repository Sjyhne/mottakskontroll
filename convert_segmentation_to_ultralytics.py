import cv2
import numpy as np
import pathlib

def convert_mask_to_yolo(mask_path, image_path, label_path, class_id=0):
    # Load mask and image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    img_height, img_width = image.shape[:2]

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare annotations
    yolo_annotations = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Save annotations
    with open(label_path, 'w') as f:
        for annotation in yolo_annotations:
            f.write(f"{annotation}\n")

# Example usage
mask_dir = 'data\\560732.1454_7017453.3362_561504.6032_7019458.9953_0.2_500_500\\masks'
image_dir = 'data\\560732.1454_7017453.3362_561504.6032_7019458.9953_0.2_500_500\\images'
label_dir = 'data\\560732.1454_7017453.3362_561504.6032_7019458.9953_0.2_500_500\\labels'

mask_dir = pathlib.Path(mask_dir)
image_dir = pathlib.Path(image_dir)
label_dir = pathlib.Path(label_dir)

label_dir.mkdir(parents=True, exist_ok=True)

for mask_filename in mask_dir.glob('*'):
    image_filename = mask_filename
    label_filename = mask_filename.parent.parent / 'labels' / mask_filename.with_suffix('.txt').name

    convert_mask_to_yolo(mask_filename, image_filename, label_filename)