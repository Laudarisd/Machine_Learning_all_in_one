import os
import cv2
import json
import logging
import random
import colorsys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from ultralytics import YOLO

CONFIG = {
    "model_path": "./models/best.pt",
    "image_folder": "./data",
    "gt_dir": "./data",
    "save_dir": "./prediction_result",
    "log_dir": "./logs",
    "conf_threshold": 0.25,
    "iou_threshold": 0.5,
    "wall_iou_threshold": 0.1,
    "smoothing_epsilon": 0.002,
    "img_size": 1024,
}


class SegmentationEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.stats = []
        self._setup_dirs()
        self._setup_logging()
        self.model = YOLO(Path(config["model_path"]).resolve())
        self.class_colors = {i: self._random_color() for i in range(100)}
        logging.info("Initialized segmentation evaluator")

    def _random_color(self) -> Tuple[int, int, int]:
        h, s, v = random.random(), 0.5 + random.random() * 0.5, 0.5 + random.random() * 0.5
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    def _setup_dirs(self):
        Path(self.config["log_dir"]).mkdir(parents=True, exist_ok=True)
        for sub in ["raw", "smoothed", "tp", "fp", "gt_fp"]:
            Path(self.config["save_dir"], sub).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        log_file = Path(self.config["log_dir"], "evaluation.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode='w', encoding='utf-8'), logging.StreamHandler()]
        )

    def _load_ground_truth(self, gt_path: str, shape: Tuple[int, int], class_names: Dict) -> Tuple[List[np.ndarray], List[int]]:
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
        except Exception as e:
            logging.error(f"Error reading GT {gt_path}: {e}")
        return gt_masks, gt_classes

    def _calculate_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        pred_bin, gt_bin = (pred > 0.5), (gt > 0)
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        iou_cal = intersection / union if union else 0.0
        return iou_cal

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.sum() < 100:
            return mask
        smoothed = np.zeros_like(mask)
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            epsilon = self.config["smoothing_epsilon"] * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                cv2.fillPoly(smoothed, [approx], 1)
        return smoothed if smoothed.sum() > 0 else mask

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
            cv2.drawContours(vis, contours, -1, (255, 0, 255), 2)
        for mask in fp_masks:
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(str(save_path), vis)

    def process_image(self, image_path: str, gt_path: str):
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Skipping unreadable image: {image_path}")
            return
        height, width = image.shape[:2]
        #print(height, width)

        results = self.model.predict(source=image, conf=self.config["conf_threshold"],
                             imgsz=self.config["img_size"], verbose=False)[0]

        #print(results)
        if results.masks is None:
            logging.info(f"No predictions for {image_path}")
            return

        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        class_names = results.names
        #print(f" Prediction classes: {class_names}")

        image_name = Path(image_path).name
        save_dir = Path(self.config["save_dir"])

        # Load ground truth
        gt_masks, gt_classes = self._load_ground_truth(gt_path, (height, width), class_names)
        #print(f" GT classes: {gt_classes}")

        self._visualize(image, masks, classes, save_dir / "raw" / f"raw_{image_name}", width, height)

        smoothed_masks = [self._smooth_mask(masks[i]) for i in range(len(masks))]
        smoothed_classes = [classes[i] for i in range(len(masks))]
        self._visualize(image, smoothed_masks, smoothed_classes, save_dir / "smoothed" / f"smoothed_{image_name}", width, height)

        tp, fp = [], []
        available_gt = list(zip(gt_masks, gt_classes))

        for i, (mask, cls) in enumerate(zip(masks, classes)):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            threshold = self.config["wall_iou_threshold"] if class_names.get(cls) == "wall" else self.config["iou_threshold"]
            best_iou, best_j = 0, -1
            for j, (gt_mask, gt_cls) in enumerate(available_gt):
                if gt_cls != cls:
                    continue
                iou = self._calculate_iou(mask, gt_mask)
                if iou > best_iou and iou >= threshold:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                tp.append(i)
                available_gt.pop(best_j)
            else:
                fp.append(i)

        if tp:
            self._visualize(image, [masks[i] for i in tp], [classes[i] for i in tp], save_dir / "tp" / f"tp_{image_name}", width, height)
        if fp:
            self._visualize(image, [masks[i] for i in fp], [classes[i] for i in fp], save_dir / "fp" / f"fp_{image_name}", width, height, False)
        if gt_masks or fp:
            self._visualize_gt_fp(image, gt_masks, [masks[i] for i in fp], save_dir / "gt_fp" / f"gt_fp_{image_name}", width, height)

        for cls_name in set(class_names.values()):
            self.stats.append({
                "image": image_name,
                "class": cls_name,
                "tp": sum(1 for i in tp if class_names.get(classes[i]) == cls_name),
                "fp": sum(1 for i in fp if class_names.get(classes[i]) == cls_name),
                "gt": sum(1 for gt_cls in gt_classes if class_names.get(gt_cls) == cls_name),
                "predictions": sum(1 for i in range(len(classes)) if class_names.get(classes[i]) == cls_name)
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
