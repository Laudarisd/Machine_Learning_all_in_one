# src/utils/data_augmentation.py
import cv2
import numpy as np
import albumentations as A
import json
from pathlib import Path
from shapely.geometry import Polygon, box

class DataAugmenter:
    def __init__(self, img_path, json_path, save_img_dir, save_json_dir, num_augmentations, augmentation_params):
        self.img_path = img_path
        self.json_path = json_path
        self.save_img_dir = save_img_dir
        self.save_json_dir = save_json_dir
        self.num_augmentations = num_augmentations
        self.augmentation_params = augmentation_params
        
        self.image = cv2.imread(str(img_path))
        if self.image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        self.json_data = self._load_json(json_path)
        self.polygons, self.labels = self._parse_polygons()
        self.original_name = Path(img_path).stem

    def _load_json(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON at {json_path}: {e}")

    def _parse_polygons(self):
        polygons = [[(p[0], p[1]) for p in shape["points"]] 
                   for shape in self.json_data["shapes"] 
                   if shape["shape_type"] == "polygon"]
        labels = [shape["label"] for shape in self.json_data["shapes"] 
                 if shape["shape_type"] == "polygon"]
        return polygons, labels

    def _clip_polygon(self, polygon, image_shape):
        try:
            poly = Polygon(polygon)
            if not poly.is_valid or poly.is_empty:
                return []
            h, w = image_shape[:2]
            boundary = box(0, 0, w, h)
            clipped = poly.intersection(boundary)
            if clipped.is_empty:
                return []
            if clipped.geom_type == 'Polygon':
                coords = list(clipped.exterior.coords)[:-1]
                return [coords] if len(coords) >= 3 else []
            elif clipped.geom_type == 'MultiPolygon':
                return [list(sub_poly.exterior.coords)[:-1] 
                       for sub_poly in clipped.geoms 
                       if len(sub_poly.exterior.coords) >= 4]
            return []
        except Exception as e:
            print(f"Clipping error: {e}")
            return []

    def get_augmentation_pipeline(self):
        transforms = []
        if self.augmentation_params.get("crop", False):
            crop_scale = self.augmentation_params.get("crop_scale", 0.9)
            crop_height = int(self.image.shape[0] * crop_scale)
            crop_width = int(self.image.shape[1] * crop_scale)
            transforms.append(A.RandomCrop(
                height=crop_height,
                width=crop_width,
                p=self.augmentation_params.get("p_crop", 0.5)
            ))
        if self.augmentation_params.get("rotate", False):
            transforms.append(A.Rotate(
                limit=self.augmentation_params.get("angle_limit", 30),
                p=self.augmentation_params.get("p_rotate", 0.5)
            ))
        if self.augmentation_params.get("flip_h", False):
            transforms.append(A.HorizontalFlip(p=self.augmentation_params.get("p_flip_h", 0.5)))
        if self.augmentation_params.get("flip_v", False):
            transforms.append(A.VerticalFlip(p=self.augmentation_params.get("p_flip_v", 0.5)))
        transforms.append(A.RandomBrightnessContrast(p=0.5))  # Additional augmentation from PolygonAugmenter
        
        return A.Compose(transforms, 
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def augment_and_save(self):
        keypoints = [pt for poly in self.polygons for pt in poly]
        sizes = [len(poly) for poly in self.polygons]
        
        for i in range(self.num_augmentations):
            transform = self.get_augmentation_pipeline()
            aug = transform(image=self.image, keypoints=keypoints)
            aug_image = aug["image"]
            aug_keypoints = aug["keypoints"]

            # Reconstruct and clip polygons
            aug_polygons = []
            aug_labels = []
            start = 0
            for j, size in enumerate(sizes):
                poly_points = aug_keypoints[start:start + size]
                clipped_polys = self._clip_polygon(poly_points, aug_image.shape)
                for clipped_poly in clipped_polys:
                    aug_polygons.append(clipped_poly)
                    aug_labels.append(self.labels[j])
                start += size

            # Skip if no valid polygons remain after clipping
            if not aug_polygons:
                print(f"Skipping augmentation {i} for {self.img_path}: No valid polygons after clipping")
                continue

            # Determine augmentation type for filename
            aug_type = []
            if self.augmentation_params.get("crop") and np.random.random() < self.augmentation_params.get("p_crop", 0.5):
                aug_type.append("crop")
            if self.augmentation_params.get("rotate") and np.random.random() < self.augmentation_params.get("p_rotate", 0.5):
                aug_type.append("rotate")
            if self.augmentation_params.get("flip_h") and np.random.random() < self.augmentation_params.get("p_flip_h", 0.5):
                aug_type.append("fliph")
            if self.augmentation_params.get("flip_v") and np.random.random() < self.augmentation_params.get("p_flip_v", 0.5):
                aug_type.append("flipv")
            aug_str = "_".join(aug_type) if aug_type else "basic"

            # Save augmented image
            aug_img_name = f"{self.original_name}_aug_{i}_{aug_str}.png"
            aug_img_path = Path(self.save_img_dir) / aug_img_name
            aug_img_path.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(aug_img_path), aug_image):
                print(f"Saved image: {aug_img_path}")
            else:
                print(f"Failed to save image: {aug_img_path}")

            # Create and save JSON in LabelMe format
            aug_json = {
                "version": "5.2.1",
                "flags": {},
                "shapes": [
                    {
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    for label, points in zip(aug_labels, aug_polygons)
                ],
                "imagePath": str(Path("..") / "images" / aug_img_name),
                "imageData": None,
                "imageHeight": aug_image.shape[0],
                "imageWidth": aug_image.shape[1]
            }
            aug_json_name = f"{self.original_name}_aug_{i}_{aug_str}.json"
            aug_json_path = Path(self.save_json_dir) / aug_json_name
            aug_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(aug_json_path, 'w', encoding='utf-8') as f:
                json.dump(aug_json, f, indent=2)
            print(f"Saved JSON: {aug_json_path}")

def augment_dataset(data_dir, json_dir, save_img_dir, save_json_dir, num_augmentations, augmentation_params):
    data_path = Path(data_dir)
    json_path = Path(json_dir)
    
    print(f"Scanning {data_path} for images...")
    for img_file in data_path.glob("*.png"):
        json_file = json_path / f"{img_file.stem}.json"
        print(f"Checking {img_file} -> {json_file}")
        if json_file.exists():
            print(f"Augmenting {img_file}...")
            augmenter = DataAugmenter(
                img_path=img_file,
                json_path=json_file,
                save_img_dir=save_img_dir,
                save_json_dir=save_json_dir,
                num_augmentations=num_augmentations,
                augmentation_params=augmentation_params
            )
            augmenter.augment_and_save()
        else:
            print(f"No JSON found for {img_file}")