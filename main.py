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



name = 'COCO_train2014_000000000081'
img = Image.open('E://Data//COCO//train2014//{}.jpg'.format(name))
with open('E://Data//COCO//labels//train2014//{}.txt'.format(name), 'r') as f:
    labels = f.readlines()
# img = img.resize((7, 7))
# labels.append('11 0.448312 0.418213 0.587969 0.970257 \n')
# intersection_over_union(labels[1].split()[1:], labels[0].split()[1:])
plot_bbox(img, labels)
