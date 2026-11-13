import pandas as pd
from typing import *
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np

class ShipDataset(Dataset):
    def __init__(self, data: pd.DataFrame,
                 path: str | Path,
                 transform: v2.Compose,
                 augment:  v2.Compose, 
                 patch: (int, int),
                 compress: (int, int)):
        
        self.data = data
        self.path = path
        self.transform = transform
        self.augment = augment
        self.patch = patch
        self.compress = compress


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        item = self.data.loc[idx]

        path_to_image = os.path.join(self.path, item['ImageId'])

        image = TF.to_tensor(Image.open(path_to_image))
        label = RLE_decoding(item['RLE'])

        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        normalizing = v2.Compose([
                        v2.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])
        image = normalizing(image)
        
        if self.compress:
            image = image[:, ::compress[0], ::compress[1]]
            label = label[:, ::compress[0], ::compress[1]]

        if self.augment:
            image, label = self.augment(image), self.augment(label)
            
        if self.patch:
            #(C, H, W) -> (C, kH, kW, H, W)
            image = image.unfold(1, self.patch[0], self.patch[0])\
                         .unfold(2, self.patch[1], self.patch[1])
            label = label.unfold(1, self.patch[0], self.patch[0])\
                         .unfold(2, self.patch[1], self.patch[1])
            
            #(C, kH, kW, H, W) -> (C, kH*kW, H, W)
            image, label = torch.flatten(image, start_dim=1, end_dim=2), \
                           torch.flatten(label, start_dim=1, end_dim=2)
            
            #(C, kH*kW, H, W) -> (kH*kW, C, H, W)
            image, label = image.permute(1, 0, 2, 3), label.permute(1, 0, 2, 3)

        
        return {'image': image, 'label': label} 
