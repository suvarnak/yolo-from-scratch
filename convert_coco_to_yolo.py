import json
import os
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, output_dir):
    """Convert COCO annotations to YOLO format"""
    
    with open(coco_json_path) as f:
        data = json.load(f)
    
    # Create output directories
    labels_dir = Path(output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image list file
    img_list = []
    
    # Map category IDs to class indices (0-based)
    categories = {cat['id']: i for i, cat in enumerate(data['categories'])}
    
    # Group annotations by image
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # Process each image
    for img in data['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_width = img['width']
        img_height = img['height']
        
        # Add to image list
        img_list.append(img_name)
        
        # Create label file
        label_file = labels_dir / f"{Path(img_name).stem}.txt"
        
        with open(label_file, 'w') as f:
            if img_id in img_annotations:
                for ann in img_annotations[img_id]:
                    # Convert COCO bbox to YOLO format
                    x, y, w, h = ann['bbox']
                    
                    # Convert to center coordinates and normalize
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Get class index
                    class_id = categories[ann['category_id']]
                    
                    # Write YOLO format: class x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Create image list file
    list_file = Path(output_dir) / f"{Path(coco_json_path).stem}.txt"
    with open(list_file, 'w') as f:
        for img_name in img_list:
            f.write(f"{img_name}\n")
    
    print(f"Converted {len(img_list)} images")
    print(f"Labels saved to: {labels_dir}")
    print(f"Image list saved to: {list_file}")

if __name__ == "__main__":
    # Convert training set
    convert_coco_to_yolo("C:/Users/suvar/workspace/FTD3/coco/annotations/instances_train2017.json", "coco/train2017")
    
    # Convert validation set
    convert_coco_to_yolo("C:/Users/suvar/workspace/FTD3/coco/annotations/instances_val2017.json", "coco/val2017")