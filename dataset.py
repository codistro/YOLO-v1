import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os


class YoloDataset(Dataset):

    def __init__(self, img_path, labels_path, S=7, B=2, C=20):

        super().__init__()

        self.img_path = img_path
        self.labels_path = labels_path
        self.S = S
        self.B = B
        self.C = C

        self.annotations = os.listdir(self.labels_path)

    def __getitem__(self, item):
        img_name = self.annotations[item].split('.')[0] + '.jpg'
        img = os.path.join(self.img_path, img_name)
        labels = os.path.join(self.labels_path, self.annotations[item])
        labels = open(labels, 'r').readlines()
        output = torch.zeros(self.S, self.S, self.B * 5 + self.C)
        for _ in labels:
            cls, x, y, w, h = _.split()
            cls, x, y, w, h = int(cls), float(x), float(y), float(w), float(h)



        img = Image.open(img)

        return img, output

    def __len__(self):
        return len(self.annotations)


img_path = 'E://Data//COCO//val2014'
labels_path = 'E://Data//COCO//labels//val2014'
dataset = YoloDataset(img_path, labels_path)
print(dataset[0])