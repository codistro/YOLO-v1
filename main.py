import torch 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from utils import intersection_over_union


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

        S = 20

        i, j = int(S * x), int(S * y)

        x_cell, y_cell = S - i, S - j

        print(x_cell, y_cell)

        '''
        Test code ends
        '''

        ax.plot(x * img_width, y * img_height, 'ro')
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



name = 'COCO_val2014_000000005635'
img = Image.open('E://Data//COCO//val2014//{}.jpg'.format(name))
with open('E://Data//COCO//labels//val2014//{}.txt'.format(name), 'r') as f:
    labels = f.readlines()
img = img.resize((20, 20))
# labels.append('11 0.448312 0.418213 0.587969 0.970257 \n')
# intersection_over_union(labels[1].split()[1:], labels[0].split()[1:])
plot_bbox(img, labels)
