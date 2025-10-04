import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
if torch.backends.mps.is_available():
    DEVICE = "mps"

# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0  # use identity loss
LAMBDA_CYCLE = 10  # use cycle loss
NUM_EPOCHS = 200
NUM_WORKERS = 4
SAVE_FREQ = 100  # save model every SAVE_FREQ epochs
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genH.pth.tar"
CHECKPOINT_GEN_Z = "genZ.pth.tar"
CHECKPOINT_CRITIC_H = "criticH.pth.tar"
CHECKPOINT_CRITIC_Z = "criticZ.pth.tar"

transform = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)