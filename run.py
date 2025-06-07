# run.py

import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import shutil
import csv
import re
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("Please install the required packages: pip install segmentation-models-pytorch albumentations")
    exit()

# --------------------------------------------------------------------------
# 1. CONSTANTS AND CONFIGURATION
# --------------------------------------------------------------------------

# --- Data Paths ---
TRAINING_DIR = './training/'
TEST_IMAGE_DIR = './test_set_images/'
IMAGE_DIR = os.path.join(TRAINING_DIR, 'images')
MASK_DIR = os.path.join(TRAINING_DIR, 'groundtruth')
SUBMISSION_FILENAME = 'submission.csv'
BEST_MODEL_PATH = 'best_model.pth'

# --- Training Settings ---
ENCODER = 'resnet34'  # A powerful, pre-trained backbone
PRETRAINED = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 40
BATCH_SIZE = 16  # Adjust based on your GPU memory
LEARNING_RATE = 0.001
IMG_SIZE = 400

# --- Submission Settings ---
PATCH_SIZE = 16
FOREGROUND_THRESHOLD = 0.25


# --------------------------------------------------------------------------
# 2. DATASET AND AUGMENTATIONS
# --------------------------------------------------------------------------

def get_training_augmentations(img_size):
    """Returns a set of heavy augmentations for the training data."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=35, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentations(img_size):
    """Returns a minimal set of augmentations for validation."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class RoadDataset(Dataset):
    """Custom PyTorch Dataset for loading images and masks."""
    def __init__(self, image_dir, mask_dir, augmentations=None):
        self.image_ids = sorted(os.listdir(image_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        mask_path = os.path.join(self.mask_dir, image_id)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask

# --------------------------------------------------------------------------
# 3. TRAINING AND SUBMISSION LOGIC
# --------------------------------------------------------------------------

def train_model():
    """Handles the entire model training process."""
    print("--- Starting Model Training ---")
    print(f"Using Device: {DEVICE}")

    # --- Model, Loss, Optimizer, and Scheduler Setup ---
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=PRETRAINED,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    ).to(DEVICE)

    loss_fn = smp.losses.TverskyLoss(mode='binary', alpha=0.5, beta=0.5)
    loss_fn += smp.losses.SoftBCEWithLogitsLoss()
    loss_fn.__name__ = 'DiceBCE_Loss'

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    metrics = [smp.utils.metrics.IoU(threshold=0.5), smp.utils.metrics.Fscore(threshold=0.5)]

    # --- Data Splitting and Loading ---
    all_image_ids = sorted(os.listdir(IMAGE_DIR))
    random.seed(42)
    random.shuffle(all_image_ids)
    
    train_size = int(0.8 * len(all_image_ids))
    train_ids = all_image_ids[:train_size]
    valid_ids = all_image_ids[train_size:]

    temp_train_img_dir = 'temp_train/images'
    temp_train_mask_dir = 'temp_train/masks'
    temp_valid_img_dir = 'temp_valid/images'
    temp_valid_mask_dir = 'temp_valid/masks'

    for d in [temp_train_img_dir, temp_train_mask_dir, temp_valid_img_dir, temp_valid_mask_dir]:
        os.makedirs(d, exist_ok=True)

    print("Copying files to temporary training and validation directories...")
    for img_id in train_ids:
        shutil.copy(os.path.join(IMAGE_DIR, img_id), os.path.join(temp_train_img_dir, img_id))
        shutil.copy(os.path.join(MASK_DIR, img_id), os.path.join(temp_train_mask_dir, img_id))
    for img_id in valid_ids:
        shutil.copy(os.path.join(IMAGE_DIR, img_id), os.path.join(temp_valid_img_dir, img_id))
        shutil.copy(os.path.join(MASK_DIR, img_id), os.path.join(temp_valid_mask_dir, img_id))

    train_dataset = RoadDataset(temp_train_img_dir, temp_train_mask_dir, augmentations=get_training_augmentations(IMG_SIZE))
    valid_dataset = RoadDataset(temp_valid_img_dir, temp_valid_mask_dir, augmentations=get_validation_augmentations(IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # --- Training Loop ---
    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss_fn, metrics=metrics, optimizer=optimizer, device=DEVICE, verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss_fn, metrics=metrics, device=DEVICE, verbose=True)

    max_fscore = 0
    for i in range(EPOCHS):
        print(f'\nEpoch: {i+1}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        scheduler.step(valid_logs[loss_fn.__name__])

        if valid_logs['fscore'] > max_fscore:
            max_fscore = valid_logs['fscore']
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Model saved! Best F1-score: {max_fscore:.4f}")
    
    return temp_train_img_dir, temp_valid_img_dir


def generate_submission():
    """Loads the best model and generates the submission.csv file."""
    print("\n--- Generating Submission File ---")
    
    # --- Load Model ---
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()
    print("Best model loaded.")

    # --- Test Transformations ---
    test_transform = A.Compose([
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def patch_to_label(patch, threshold):
        return 1 if np.mean(patch) > threshold else 0

    # --- File Discovery and Prediction Loop ---
    test_image_paths = []
    for root, _, files in os.walk(TEST_IMAGE_DIR):
        for file in files:
            if file.endswith('.png'):
                test_image_paths.append(os.path.join(root, file))
    test_image_paths.sort()

    with open(SUBMISSION_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'prediction'])

        for image_path in tqdm(test_image_paths, desc="Creating Submission"):
            parent_dir_name = os.path.basename(os.path.dirname(image_path))
            img_number_match = re.search(r'(\d+)', parent_dir_name)
            if not img_number_match:
                continue
            img_number = int(img_number_match.group(1))

            image = cv2.imread(image_path)
            original_h, original_w, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            transformed = test_transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_mask = model(input_tensor)
                pred_mask_padded = pred_mask.squeeze().cpu().numpy()
                pred_mask_unpadded = pred_mask_padded[:original_h, :original_w]
                binary_mask = (pred_mask_unpadded > 0.5).astype(np.uint8)

            for y in range(0, original_h, PATCH_SIZE):
                for x in range(0, original_w, PATCH_SIZE):
                    patch = binary_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    label = patch_to_label(patch, FOREGROUND_THRESHOLD)
                    submission_id = "{:03d}_{}_{}".format(img_number, y, x)
                    writer.writerow([submission_id, label])

    print(f"\n--- Submission file '{SUBMISSION_FILENAME}' created successfully! ---")


# --------------------------------------------------------------------------
# 4. SCRIPT EXECUTION
# --------------------------------------------------------------------------

if __name__ == '__main__':
    temp_dirs_to_remove = []
    try:
        # Run the training process
        temp_train_dir, temp_valid_dir = train_model()
        temp_dirs_to_remove.extend([os.path.dirname(temp_train_dir), os.path.dirname(temp_valid_dir)])
        
        # Run the submission generation process
        generate_submission()
    finally:
        # Clean up temporary directories
        if temp_dirs_to_remove:
            print("\nCleaning up temporary directories...")
            for d in temp_dirs_to_remove:
                if os.path.exists(d):
                    shutil.rmtree(d)
            print("Cleanup complete.")