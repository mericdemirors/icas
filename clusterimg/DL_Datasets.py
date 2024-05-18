import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = os.path.abspath(root_dir)
        self.x = os.listdir(self.root_dir)
        self.num_samples = len(self.x)

    def __len__(self):
        return self.num_samples

    def read_image(self, image_path):
        image = image = cv2.imread(image_path).astype(np.float32)/255
        image = np.moveaxis(image, 2, 0)
        return image

    def __getitem__(self, idx):
        image_name = self.x[idx]
        image_path = os.path.join(self.root_dir, image_name)
        
        image = self.read_image(image_path)

        return image_path, image