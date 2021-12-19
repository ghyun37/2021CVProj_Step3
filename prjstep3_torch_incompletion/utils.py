import os
import cv2
import numpy as np
import torch
from PIL import Image
from skimage import io
from torchvision import transforms
from torch.utils.data import DataLoader

convert = transforms.ToTensor()
        
class Dataset(object):
    def __init__(self, data_paths, classes, transform=None):
        self.paths = data_paths
        self.labels = self.get_label()
        self.classes = classes
        self.transform = transform
        
    def get_label(self):
        label = []
        for path in self.paths:
            # 폴더명 = 클래스 이름
            label.append(path.split('\\')[-2])
            # print(path.split('\\')[-2])
        return label
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        image = Image.open(self.paths[index])
        image = np.array(image) / 255.
        image = convert(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.labels[index])
    