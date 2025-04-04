# main.py
import argparse
from pathlib import Path

# Import necessary modules
from src.utils.json_to_yolo import process_dataset
from src.core.train import TrainYolo
from src.utils.data_augmentation import augment_dataset  # No globals needed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument('--aug', choices=['yes', 'no'], default='yes',
                        help='Whether to perform data augmentation (yes/no)')
    parser.add_argument('--root-dir', type=str, default='./raw_dataset',
                        help='Root directory containing train/test/valid folders')
    parser.add_argument('--output-dir', type=str, default='./train_data',
                        help='Output directory for processed training data')
    parser.add_argument('--model-path', type=str, default='./src/pretrain_models/yolov8l-seg.pt',
                        help='Path to the pre-trained YOLO model')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Image size for training')
    parser.add_argument('--device', type=str, default='0',
                        help='Device for training (e.g., "0" for GPU 0, "cpu" for CPU)')
    return parser.parse_args()

def configure_augmentation():
    augmentation_params = {
        "crop": True,
        "crop_scale": 0.9,
        "rotate": True,
        "angle_limit": 10,
        "flip_h": True,
        "flip_v": False,
        "p_crop": 0.5,
        "p_rotate": 0.5,
        "p_flip_h": 0.5,
        "p_flip_v": 0.5
    }
    num_augmentations = 5  # Number of augmentations to generate for each image
    return augmentation_params, num_augmentations

def main():
    args = parse_args()
    
    # Set up paths
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configure augmentation if enabled
    if args.aug == 'yes':
        augmentation_params, num_augmentations = configure_augmentation()
        for split in ['train']:
            data_dir = str(root_dir / split / 'images')
            json_dir = str(root_dir / split / 'labels')
            save_img_dir = data_dir  # Save in same directory as original
            save_json_dir = json_dir  # Save in same directory as original
            
            if Path(data_dir).exists() and Path(json_dir).exists():
                print(f"Running augmentation for {split} split...")
                augment_dataset(
                    data_dir=data_dir,
                    json_dir=json_dir,
                    save_img_dir=save_img_dir,
                    save_json_dir=save_json_dir,
                    num_augmentations=num_augmentations,
                    augmentation_params=augmentation_params
                )
            else:
                print(f"Skipping {split} - directories not found: data_dir={Path(data_dir).exists()}, json_dir={Path(json_dir).exists()}")

    # Convert to YOLO format and prepare training data
    print("Converting JSON to YOLO format...")
    process_dataset(args.root_dir, args.output_dir)
    
    # Train the model
    data_yaml_path = output_dir / 'data.yaml'
    if data_yaml_path.exists():
        if not Path(args.model_path).exists():
            print(f"Error: Model path {args.model_path} not found")
            return
        print("Starting YOLO training...")
        trainer = TrainYolo(
            data_yaml_path=str(data_yaml_path),
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=[int(d) for d in args.device.split(',')] if args.device != 'cpu' else 'cpu'
        )
        trainer.train_model()
    else:
        print(f"Error: data.yaml not found at {data_yaml_path}")

if __name__ == '__main__':
    main()
