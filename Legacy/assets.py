import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageOps
import random

class ImgDataset(Dataset):
    def __init__(self, 
                 auth_path, 
                 forged_path, 
                 mask_path, 
                 img_shape=(224, 224),
                 transform=False):

        self.x0 = [os.path.join(auth_path, f) for f in os.listdir(auth_path)]
        self.x1 = [os.path.join(forged_path, f) for f in os.listdir(forged_path)]
        self.y0 = [None] * len(self.x0)
        self.y1 = [os.path.join(mask_path, f) for f in os.listdir(mask_path)]
        
        while len(self.x0) < len(self.x1):
            i = random.randint(0, len(self.x0)-1)
            self.x0 += [self.x0[i]]
            self.y0 += [None]

        while len(self.x1) < len(self.x0):
            i = random.randint(0, len(self.x0)-1)
            self.x1 += [self.x1[i]]
            self.y1 += [self.y1[i]]
        
        self.x = self.x0 + self.x1
        self.y = self.y0 + self.y1
        
        self.H, self.W = img_shape
        self.transform = transform
        self.colorjitter = T.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.jpeg = T.JPEG(quality=(5, 100))
        self.to_tensor = transforms.ToTensor()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = Image.open(self.x[idx]).convert('RGB')
        #y = Image.open(self.x[idx]).convert('L')
        if self.y[idx]:
            y = np.any(np.load(self.y[idx]), axis=0).astype(int)
        else:
            y = np.zeros((self.H, self.W))
        y = Image.fromarray(y.astype(np.float32))

        x = ImageOps.pad(x, (self.H, self.W), color=(0, 0, 0))
        y = ImageOps.pad(y, (self.H, self.W), color=(0))

        x = self.to_tensor(x)
        y = self.to_tensor(y)

        # -------------------------------

        # Transformations (if applicable)
        if self.transform:
            x = (self.jpeg((x * 255).to(torch.uint8))/255).to(torch.float32)
            x = self.colorjitter(x)
            
            if torch.rand(1) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

            if torch.rand(1) > 0.5: 
                x = TF.vflip(x)
                y = TF.vflip(y)
         
        # -------------------------------

        return x, (y>0.5).to(torch.float32)

class ImgDataset_Classification(Dataset):
    def __init__(self, 
                 auth_path, 
                 forged_path, 
                 mask_path, 
                 img_shape=(224, 224),
                 transform=False):

        self.x0 = [os.path.join(auth_path, f) for f in os.listdir(auth_path)]
        self.x1 = [os.path.join(forged_path, f) for f in os.listdir(forged_path)]

        while len(self.x0) < len(self.x1):
            self.x0 += [self.x0[random.randint(0, len(self.x0)-1)]]

        while len(self.x1) < len(self.x0):
            self.x1 += [self.x1[random.randint(0, len(self.x1)-1)]]

        self.x = self.x0 + self.x1
        y0 = [torch.tensor([1., 0.])] * len(self.x0)
        y1 = [torch.tensor([0., 1.])] * len(self.x1)
        self.y = y0 + y1
        self.H, self.W = img_shape
        self.transform = transform
        self.colorjitter = T.RandomPhotometricDistort()
        self.crop = transforms.RandomResizedCrop(size=img_shape, scale=(0.9, 1.0))
        self.rotation = T.RandomRotation(degrees=(-15, 45))
        self.to_tensor = transforms.ToTensor()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = Image.open(self.x[idx]).convert('RGB')
        x = ImageOps.pad(x, (self.H, self.W), color=(0, 0, 0))
        x = self.to_tensor(x)
        y = self.y[idx]

        # -------------------------------

        # Transformations (if applicable)
        if self.transform:
            x = self.colorjitter(x)
            x = self.rotation(x)
            
            if torch.rand(1) > 0.5:
                x = TF.hflip(x)

            if torch.rand(1) > 0.5: 
                x = TF.vflip(x)

        # -------------------------------

        return x, y

# loss

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, iou_weight=1.0, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()

    def iou_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return 1 - iou.mean()

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        dice = self.dice_loss(preds, targets)
        iou = self.iou_loss(preds, targets)

        loss = (
            self.bce_weight * bce +
            self.dice_weight * dice +
            self.iou_weight * iou
        )
        return loss

