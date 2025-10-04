from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NoFireFireDataset(Dataset):
    def __init__(self, root_nofire, root_fire, transform=None):
        self.root_nofire = root_nofire
        self.root_fire = root_fire
        self.transform = transform
        
        self.nofire_images = os.listdir(root_nofire)
        self.fire_images = os.listdir(root_fire)
        self.nofire_len = len(self.nofire_images)
        self.fire_len = len(self.fire_images)
        self.length_dataset = max(self.nofire_len, self.fire_len)
        
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        nofire_img = self.nofire_images[index % len(self.nofire_images)]
        fire_img = self.fire_images[index % len(self.fire_images)]
        nofire_path = os.path.join(self.root_nofire, nofire_img)
        fire_path = os.path.join(self.root_fire, fire_img)
        
        nofire_img = np.array(Image.open(nofire_path).convert("RGB"))
        fire_img = np.array(Image.open(fire_path).convert("RGB"))
        
        if self.transform:
            augmentations = self.transform(image=fire_img, image0=nofire_img)
            fire_img = augmentations["image"]
            nofire_img = augmentations["image0"]
        
        return fire_img, nofire_img