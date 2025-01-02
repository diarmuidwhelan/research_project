import os
from pathlib import Path
import yaml
from google.colab import drive
import shutil
import torch
import albumentations as A
import cv2
import numpy as np

# class BallDetectionOptimiser:
#     """Enhanced settings and augmentations for ball detection"""
#     @staticmethod
#     def get_ball_augmentation():
#         """Custom augmentation pipeline for ball detection"""
#         return A.Compose([
#             # Motion blur simulation
#             A.OneOf([
#                 A.MotionBlur(blur_limit=7, p=0.5),
#                 A.GaussianBlur(blur_limit=3, p=0.3),
#             ], p=0.5),

#             # Visibility enhancement
#             A.OneOf([
#                 A.RandomBrightnessContrast(
#                     brightness_limit=0.2,
#                     contrast_limit=0.2,
#                     p=0.5
#                 ),
#                 A.HueSaturationValue(
#                     hue_shift_limit=10,
#                     sat_shift_limit=20,
#                     val_shift_limit=20,
#                     p=0.5
#                 ),
#             ], p=0.5),

#             # Small object enhancement
#             A.OneOf([
#                 A.RandomScale(scale_limit=0.2, p=0.5),
#                 A.ShiftScaleRotate(
#                     shift_limit=0.1,
#                     scale_limit=0.2,
#                     rotate_limit=15,
#                     p=0.5
#                 ),
#             ], p=0.5),
#         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def setup_dataset():
    """ dataset setup with memory efficiency"""
    drive.mount('/content/drive')
    base_dir = Path('/content/gaelic_football')
    base_dir.mkdir(exist_ok=True)

    # Create directories efficiently
    splits = ['train', 'valid', 'test']
    [[((base_dir / split) / folder).mkdir(parents=True, exist_ok=True)
      for folder in ['images', 'labels']] for split in splits]

    return str(base_dir)

def create_yaml(base_dir):
    """Create optimised dataset configuration"""
    yaml_content = {
        'path': base_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',

        'names': {
            0: 'ball',
            1: 'goalkeeper',
            2: 'player',
            3: 'referee',
            4: 'umpire'
        },
        'nc': 5,

        # Additional configurations for ball detection
        'anchors': {
            'small': [4, 8, 12],     # Small anchors for ball
            'medium': [16, 32, 64],   # Medium anchors
            'large': [128, 256, 512]  # Large anchors
        },

        # Class weights to focus on ball
        'class_weights': {
            '0': 2.0,    # Higher weight for ball
            '1': 1.0,    # Normal weight for others
            '2': 1.0,
            '3': 1.0,
            '4': 1.0
        }
    }

    yaml_path = Path(base_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    return str(yaml_path)

def copy_dataset_from_drive():
    """Memory-efficient dataset copying"""
    drive_dataset_path = '/content/drive/MyDrive/cvdata-2'
    colab_dataset_path = '/content/gaelic_football'

    if os.path.exists(drive_dataset_path):
        print("Copying dataset efficiently...")
        for split in ['train', 'valid', 'test']:
            for folder in ['images', 'labels']:
                src = os.path.join(drive_dataset_path, split, folder)
                dst = os.path.join(colab_dataset_path, split, folder)
                if os.path.exists(src):
                    # Use system copy for memory efficiency
                    os.system(f'cp -r "{src}"/* "{dst}/"')

def train_model(yaml_path):
    """ training configuration for ball detection"""
    from ultralytics import YOLO

    # Configure CUDA for memory efficiency
    torch.cuda.empty_cache()

    # Initialise model
    model = YOLO('yolov8m.pt')

    #  training configuration
    results = model.train(
        data=yaml_path,
        epochs=40,            # Extended training
        imgsz=640,           # Standard size
        batch=8,             # Conservative batch size

        # Optimiser settings
        optimizer='AdamW',
        lr0=0.001,          # Initial learning rate
        lrf=0.0001,         # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        # Enhanced loss weights
        box=10.0,           # Increased box loss for better localisation
        cls=0.3,            # Reduced classification loss
        dfl=2.0,            # Increased DFL loss

        # Ball-focused augmentation
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.4,     # Enhanced copy-paste for balls
        degrees=10.0,
        translate=0.2,
        scale=0.7,
        shear=2.0,
        perspective=0.0001,
        flipud=0.5,
        fliplr=0.5,

        # Colour augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Additional parameters
        overlap_mask=True,   # Enhanced mask overlap
        mask_ratio=4,        # Mask processing ratio

        # Output configuration
        project='/content/gaelic_football_results',
        name='yolov8m_ball_enhanced',
        exist_ok=True,

        # Memory optimisation
        cache=False,         # Disable caching
        workers=4,           # Reduced workers

        # Device settings
        device=0,           # Use GPU
        half=True          # Use FP16
    )

    return model, results

def verify_dataset_structure(base_dir):
    """ dataset verification"""
    base_dir = Path(base_dir)
    dataset_stats = {}

    for split in ['train', 'valid', 'test']:
        split_stats = {}
        for folder in ['images', 'labels']:
            dir_path = base_dir / split / folder
            if dir_path.exists():
                files = list(dir_path.glob('*'))
                split_stats[folder] = len(files)

                # Verify ball annotations
                if folder == 'labels':
                    ball_count = 0
                    for f in files:
                        with open(f, 'r') as label_file:
                            ball_count += sum(1 for line in label_file if line.startswith('0'))
                    split_stats['ball_annotations'] = ball_count

        dataset_stats[split] = split_stats
        print(f"\n{split} set statistics:")
        print(f"Images: {split_stats.get('images', 0)}")
        print(f"Labels: {split_stats.get('labels', 0)}")
        print(f"Ball annotations: {split_stats.get('ball_annotations', 0)}")

    return dataset_stats

def main():
    """ main execution pipeline"""
    try:
        print("Setting up dataset structure...")
        base_dir = setup_dataset()

        print("\nCopying dataset from Drive...")
        copy_dataset_from_drive()

        print("\nVerifying dataset structure...")
        dataset_stats = verify_dataset_structure(base_dir)

        print("\nCreating dataset configuration...")
        yaml_path = create_yaml(base_dir)

        print("\nStarting training...")
        model, results = train_model(yaml_path)

        return model, results, dataset_stats

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure CUDA memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True

    # Run pipeline
    model, results, stats = main()
