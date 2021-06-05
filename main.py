import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from utils import intersection_over_union
from dataset import YoloDataset
from loss import YoloLoss


def plot_bbox(img, labels):
    
    fig, ax = plt.subplots()
    ax.imshow(img)

    img_width = img.width
    img_height = img.height

    for _ in labels:
        cls, x, y, w, h = _.split()
        x, y, w, h = float(x), float(y), float(w), float(h)

        '''
        Test code
        '''
        # print(x, y, w, h)
        # S = 7
        #
        # i, j = int(S * x), int(S * y)
        #
        # i_cell, j_cell = S * x - i, S * y - j
        #
        # print(i, j)
        # print(i_cell, j_cell)
        # print(S * w, S * h)

        '''
        Test code ends
        '''

        #ax.plot(x * img_width, y * img_height, 'ro')
        x = x - w/2
        y = y - h/2

        ax.add_patch(patches.Rectangle(
            (x * img_width, y * img_height),
            w * img_width,
            h * img_height,
            linewidth=3,
            edgecolor='b',
            facecolor='none'
        ))
        
    plt.grid()
    plt.show()
