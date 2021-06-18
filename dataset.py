import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


class YoloDataset(Dataset):

    def __init__(self, img_path, labels_path, S=7, B=2, C=80, transform=None):

        super().__init__()

        self.img_path = img_path
        self.labels_path = labels_path
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

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

            i, j = int(self.S * x), int(self.S * y)
            # print(i, j)
            x_cell, y_cell = self.S * x - i, self.S * y - j
            # print(x_cell, y_cell)
            w_grid, h_grid = self.S * w, self.S * h
            # print(w_grid, h_grid)
            if output[i, j, 0] == 0.0:

                output[i, j, 0] = 1
                output[i, j, 1] = x_cell
                output[i, j, 2] = y_cell
                output[i, j, 3] = w_grid
                output[i, j, 4] = h_grid

                output[i, j, cls + 10] = 1

            elif output[i, j, 5] == 0:

                output[i, j, 5] = 1
                output[i, j, 6] = x_cell
                output[i, j, 7] = y_cell
                output[i, j, 8] = w_grid
                output[i, j, 9] = h_grid

                output[i, j, cls + 10] = 1

            else:
                pass
                #print('Ignoring BBOX for image {} as only 2 is allowed per cell'.format(img_name))

        img = Image.open(img)
        if transforms:
            img = self.transform(img)

        return img, output

    def __len__(self):
        return len(self.annotations)
