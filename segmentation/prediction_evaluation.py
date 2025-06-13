from collections import defaultdict
import os
import cv2
import json
import time
import logging
import random
import colorsys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from ultralytics import YOLO
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import explain_validity


CONFIG = {
    # === Model Settings ===
    "model_path": "./models/best.pt",
    "conf_threshold": 0.25,
    "img_size": 1024,

    # === Input/Output Paths ===
    "image_folder": "./data",
    "gt_dir": "./data",
    "save_dir": "./prediction_result",
    "log_dir": "./logs",

    # === Evaluation Settings ===
    "iou_threshold": 0.5,
    "smoothing_epsilon": 0.002,
    "mask_binarize_threshold": 0.5,  # Threshold to binarize prediction masks before IoU
    "smooth_mask_threshold": 0.5,  # Threshold to binarize masks before contour smoothing


    
    #=== Evaluate Desired Classes ===
    "eval_classes": [] # Empty means evaluate all
}



class SegmentationEvaluator:
    """Evaluates segmentation model predictions against ground truth using IoU and visualizes results."""
    def __init__(self, config: Dict):
        """Initializes the evaluator with configuration, model, directories, and logging."""
        self.config = config
        self.stats = []
        self._setup_dirs()
        self._setup_logging()
        self.model = YOLO(Path(config["model_path"]).resolve())
        self.class_colors = {i: self._random_color() for i in range(100)}
        logging.info("Initialized segmentation evaluator")

    def _random_color(self) -> Tuple[int, int, int]:
        """Generates a random RGB color using HSV space."""
        h, s, v = random.random(), 0.5 + random.random() * 0.5, 0.5 + random.random() * 0.5
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    def _setup_dirs(self):
        """Creates necessary directories for saving logs and visualizations."""
        Path(self.config["log_dir"]).mkdir(parents=True, exist_ok=True)
        for sub in ["raw", "smoothed", "tp", "fp", "gt_fp"]:
            Path(self.config["save_dir"], sub).mkdir(parents=True, exist_ok=True)
            logging.info(f"Desired Directories are created.")

    def _setup_logging(self):
        """Initializes logging to file and console."""
        log_file = Path(self.config["log_dir"], "evaluation.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True  # <-- This line ensures it overrides any existing logging setup
        )

    def _load_ground_truth(self, gt_path: str, shape: Tuple[int, int], class_names: Dict) -> Tuple[List[np.ndarray], List[int]]:
        """Loads ground truth masks and classes from a JSON file and rasterizes them."""
        gt_masks, gt_classes = [], []
        class_to_id = {name.lower(): idx for idx, name in class_names.items()}
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                shapes = json.load(f).get('shapes', [])
            for s in shapes:
                label = s.get('label', '').lower()
                points = s.get('points', [])
                if label not in class_to_id or len(points) < 3:
                    continue
                pts = np.clip(np.array(points, np.float32), [0, 0], [shape[1], shape[0]])
                mask = np.zeros(shape[:2], np.uint8)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                if mask.sum() >= 100:
                    gt_masks.append(mask)
                    gt_classes.append(class_to_id[label])
            #logging.info(f"Classes from ground truth datal {gt_classes}")
        except Exception as e:
            logging.error(f"Error reading GT {gt_path}: {e}")
        return gt_masks, gt_classes

    def _calculate_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculates the Intersection over Union (IoU) between prediction and ground truth masks.
        Measures overlap quality between two masks using a configurable binarization threshold."""
        threshold = self.config.get("mask_binarize_threshold", 0.5)
        pred_bin, gt_bin = (pred > threshold), (gt > 0)
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        iou_cal = intersection / union if union else 0.0
        return iou_cal

    def _is_wall(self, class_id: int) -> bool:
        """Check if the class ID corresponds to 'wall' class"""
        return self.model.names.get(class_id, "").lower() == 'wall'

    def _smooth_mask(self, masks: List[np.ndarray], classes: List[int], confidences: List[float]) -> Tuple[List[np.ndarray], List[int]]:
        threshold = self.config.get("smooth_mask_threshold", 0.5)
        epsilon_ratio = self.config["smoothing_epsilon"]
        erosion_kernel = np.ones((3, 3), np.uint8)

        polygons_with_meta = []

        # === Step 1: Extract valid polygons from each mask ===
        for idx, (mask, cls, conf) in enumerate(zip(masks, classes, confidences)):
            if not self._is_wall(cls):
                mask = cv2.erode(mask.astype(np.uint8), erosion_kernel, iterations=2)

            bin_mask = (mask > threshold).astype(np.uint8)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c_idx, cnt in enumerate(contours):
                if len(cnt) < 3:
                    continue

                epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                coords = approx.squeeze()

                if coords.ndim != 2 or coords.shape[0] < 3:
                    continue

                # Ensure closed polygon
                if not np.array_equal(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])

                poly = Polygon(coords)

                # Try to fix invalid polygon
                if not poly.is_valid:
                    fixed = poly.buffer(0)
                    if fixed.is_valid and fixed.area > 0:
                        print(f"[DEBUG] Fixed polygon #{c_idx} with buffer(0): area={fixed.area:.2f}")
                        poly = fixed
                    else:
                        reason = explain_validity(poly)
                        print(f"[DEBUG] Skipping invalid polygon #{c_idx}: reason={reason}")
                        continue

                polygons_with_meta.append({
                    "polygon": poly,
                    "class": cls,
                    "confidence": conf
                })

        final_masks = []
        final_classes = []

        # === Step 2: Merge polygons within same class or wall class ===
        used = [False] * len(polygons_with_meta)
        merged_polygons = []

        for i, entry_i in enumerate(polygons_with_meta):
            if used[i]:
                continue

            group = [entry_i]
            used[i] = True
            poly_i = entry_i["polygon"]
            cls_i = entry_i["class"]

            for j in range(i + 1, len(polygons_with_meta)):
                if used[j]:
                    continue

                entry_j = polygons_with_meta[j]
                poly_j = entry_j["polygon"]
                cls_j = entry_j["class"]

                if (cls_i == cls_j) or (self._is_wall(cls_i) and self._is_wall(cls_j)):
                    inter = poly_i.intersection(poly_j)
                    if inter.area > 0 and inter.area / min(poly_i.area, poly_j.area) > 0.9:
                        group.append(entry_j)
                        used[j] = True

            # Merge all in the group
            merged_poly = unary_union([e["polygon"] for e in group])
            if merged_poly.is_empty or not merged_poly.is_valid:
                print(f"[DEBUG] Skipping invalid merged polygon at group {i}")
                continue

            # Use largest if it's MultiPolygon
            if isinstance(merged_poly, MultiPolygon):
                merged_poly = max(merged_poly.geoms, key=lambda g: g.area)

            merged_polygons.append({
                "polygon": merged_poly,
                "class": cls_i,
                "confidence": entry_i["confidence"]
            })


            # === Step 4: Smooth polygon ===
            components = list(merged_poly.geoms) if isinstance(merged_poly, MultiPolygon) else [merged_poly]

            for k, geom in enumerate(components):
                if not geom.is_valid or geom.is_empty or geom.area <= 0:
                    continue
                cnt = np.array(geom.exterior.coords, dtype=np.int32)
                epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:
                    smooth_poly = Polygon(approx.squeeze())
                    if not smooth_poly.is_valid:
                        smooth_poly = smooth_poly.buffer(0)
                    if smooth_poly.is_valid and smooth_poly.area > 0:
                        mask_out = np.zeros_like(masks[0], dtype=np.uint8)
                        coords = np.array(smooth_poly.exterior.coords, dtype=np.int32)
                        cv2.fillPoly(mask_out, [coords], 1)
                        final_masks.append(mask_out)
                        final_classes.append(cls_i)
                else:
                    print(f"[DEBUG] Smoothed polygon from group {i} is invalid or empty")


        return final_masks, final_classes


    def _visualize(self, image: np.ndarray, masks: List[np.ndarray], classes: List[int],
                   save_path: str, width: int, height: int, color_by_class=True):
        vis = image.copy()
        for mask, cls in zip(masks, classes):
            if mask.sum() < 100:
                continue
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            color = self.class_colors.get(cls, (255, 255, 255)) if color_by_class else (255, 0, 0)
            vis[mask == 1] = vis[mask == 1] * 0.6 + np.array(color) * 0.4
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)
        cv2.imwrite(str(save_path), vis)

    def _visualize_gt_fp(self, image, gt_masks, fp_masks, save_path, width, height):
        vis = image.copy()
        for mask in gt_masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
        for mask in fp_masks:
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(str(save_path), vis)

    def process_image(self, image_path: str, gt_path: str):
        overall_start = time.time()
        t0 = time.time()

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Skipping unreadable image: {image_path}")
            return

        height, width = image.shape[:2]
        print(f"[TIMER] Image loading took {time.time() - t0:.2f} sec")
        
        results = self.model.predict(
            source=image,
            conf=self.config["conf_threshold"],
            imgsz=self.config["img_size"],
            verbose=False
        )[0]

        print(f"[TIMER] Model prediction took {time.time() - t0:.2f} sec")

        if results.masks is None:
            logging.info(f"No predictions for {image_path}")
            return
        t0 = time.time()
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        class_names = results.names
        print(f"[TIMER] Results extraction took {time.time() - t0:.2f} sec")

        image_name = Path(image_path).name
        save_dir = Path(self.config["save_dir"])

        t0 = time.time()
        # Load GT once
        gt_masks, gt_classes = self._load_ground_truth(gt_path, (height, width), class_names)
        print(f"[TIMER] Ground truth loading took {time.time() - t0:.2f} sec")

        t0 = time.time()

        # Visualize raw predictions
        self._visualize(image, masks, classes, save_dir / "raw" / f"raw_{image_name}", width, height)
        print(f"[TIMER] Raw visualization took {time.time() - t0:.2f} sec")
    
    
        
        t0 = time.time()
        # Smooth predictions
        smoothed_masks, smoothed_classes = self._smooth_mask(masks, classes, confidences)
        print(f"[TIMER] Mask smoothing took {time.time() - t0:.2f} sec")

        t0 = time.time()
        # Visualize smoothed predictions
        self._visualize(image, smoothed_masks, smoothed_classes, save_dir / "smoothed" / f"smoothed_{image_name}", width, height)
        print(f"[TIMER] Smoothed visualization took {time.time() - t0:.2f} sec")


        t0 = time.time()
        # Evaluate predictions
        tp, fp = [], []
        available_gt = list(zip(gt_masks, gt_classes))
        gt_by_class = defaultdict(list)
        for gt_mask, gt_cls in available_gt:
            gt_by_class[gt_cls].append(gt_mask)

        for i, (mask, cls) in enumerate(zip(smoothed_masks, smoothed_classes)):
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)

            threshold = self.config["iou_threshold"]
            best_iou, best_j = 0, -1

            for j, gt_mask in enumerate(gt_by_class.get(cls, [])):
                iou = self._calculate_iou(mask, gt_mask)
                if iou > best_iou and iou >= threshold:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                tp.append(i)
                del gt_by_class[cls][best_j]
            else:
                fp.append(i)
        print(f"[TIMER] TP/FP matching took {time.time() - t0:.2f} sec")

        # Visualize TP, FP, and GT vs FP
        if tp:
            self._visualize(image, [smoothed_masks[i] for i in tp], [smoothed_classes[i] for i in tp], save_dir / "tp" / f"tp_{image_name}", width, height)
        if fp:
            self._visualize(image, [smoothed_masks[i] for i in fp], [smoothed_classes[i] for i in fp], save_dir / "fp" / f"fp_{image_name}", width, height)
        if gt_masks or fp:
            self._visualize_gt_fp(image, gt_masks, [smoothed_masks[i] for i in fp], save_dir / "gt_fp" / f"gt_fp_{image_name}", width, height)

        # Stats collection
        for cls_name in set(class_names.values()):
            self.stats.append({
                "image": image_name,
                "class": cls_name,
                "tp": sum(1 for i in tp if class_names.get(smoothed_classes[i]) == cls_name),
                "fp": sum(1 for i in fp if class_names.get(smoothed_classes[i]) == cls_name),
                "gt": sum(1 for gt_cls in gt_classes if class_names.get(gt_cls) == cls_name),
                "predictions": sum(1 for i in range(len(smoothed_classes)) if class_names.get(smoothed_classes[i]) == cls_name)
            })



    def run(self):
        image_files = [
            (Path(root) / f, Path(self.config["gt_dir"]) / f"{Path(f).stem}.json")
            for root, _, files in os.walk(self.config["image_folder"])
            for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG'))
        ]
        valid_files = [(i, g) for i, g in image_files if g.exists()]
        print(valid_files)

        if not valid_files:
            logging.warning("No valid image-GT pairs found.")
            return

        for img, gt in valid_files:
            self.process_image(str(img), str(gt))

        df = pd.DataFrame(self.stats)
        csv_path = Path(self.config["save_dir"]) / "stats.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved results to {csv_path}")


if __name__ == "__main__":
    evaluator = SegmentationEvaluator(CONFIG)
    evaluator.run()
    logging.info("Evaluation completed successfully.")

