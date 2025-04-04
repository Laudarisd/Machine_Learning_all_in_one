# src/utils/json_to_yolo.py
import json
import os
from glob import glob
from pathlib import Path
import shutil
import yaml

def normalize_coordinates(points, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y in points]

def extract_unique_classes(json_files):
    unique_labels = set()
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                unique_labels.add(shape['label'])
    return {label: idx for idx, label in enumerate(sorted(unique_labels))}

def convert_labelme_to_yolo(json_path, output_dir, class_map):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    filename = Path(data['imagePath']).stem
    output_path = Path(output_dir) / f"{filename}.txt"
    
    with open(output_path, 'w') as f:
        for shape in data['shapes']:
            if shape['shape_type'] != 'polygon':
                continue
            label = shape['label']
            if label not in class_map:
                print(f"Warning: Label '{label}' not in class map. Skipping in {json_path}")
                continue
            class_id = class_map[label]
            points = shape['points']
            
            normalized_points = normalize_coordinates(points, img_width, img_height)
            flattened_points = [coord for point in normalized_points for coord in point]
            line = f"{class_id} " + " ".join(map(str, flattened_points))
            f.write(line + '\n')
    
    return output_path

def create_yaml(class_map, output_dir):
    yaml_data = {
        'path': str(Path(output_dir).absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_map),
        'names': list(class_map.keys())
    }
    with open(Path(output_dir) / 'data.yaml', 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

def process_dataset(root_dir, output_dir):
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all JSON files to extract classes
    all_json_files = list(root_path.glob('**/*.json'))
    class_map = extract_unique_classes(all_json_files)
    print(f"Found {len(class_map)} unique classes: {class_map}")
    
    # Process each split (train, test, valid)
    for split in ['train', 'test', 'valid']:
        input_img_dir = root_path / split / 'images'
        input_json_dir = root_path / split / 'labels'
        
        if not input_img_dir.exists() or not input_json_dir.exists():
            print(f"Skipping {split} - directory not found")
            continue
            
        output_img_dir = output_path / split / 'images'
        output_label_dir = output_path / split / 'labels'
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        json_files = list(input_json_dir.glob('*.json'))
        for json_file in json_files:
            try:
                # Find corresponding image
                img_name = Path(json_file).stem + '.png'
                img_path = input_img_dir / img_name
                
                if not img_path.exists():
                    print(f"Image not found for {json_file}")
                    continue
                    
                # Convert to YOLO format
                convert_labelme_to_yolo(json_file, output_label_dir, class_map)
                
                # Copy image
                shutil.copy(img_path, output_img_dir / img_name)
                
                print(f"Processed {json_file}")
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
    
    # Create YAML file
    create_yaml(class_map, output_dir)
    print(f"YAML file created at {output_dir}/data.yaml")