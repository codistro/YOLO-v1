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



name = 'COCO_train2014_000000007143'
img = Image.open('E://Data//COCO//train2014//{}.jpg'.format(name))
with open('E://Data//COCO//labels//train2014//{}.txt'.format(name), 'r') as f:
    labels = f.readlines()
# img = img.resize((7, 7))
# labels.append('11 0.448312 0.418213 0.587969 0.970257 \n')
# intersection_over_union(labels[1].split()[1:], labels[0].split()[1:])


img_path = 'E://Data//COCO//val2014'
labels_path = 'E://Data//COCO//labels//val2014'
transform = transforms.Compose([
    transforms.Resize((24, 24)),
    transforms.ToTensor()
])
dataset = YoloDataset(img_path, labels_path, transform=transform)

from torch.utils.data import DataLoader

train_data = DataLoader(dataset, batch_size=32, drop_last=True)

idx = 0
for i, label in enumerate(dataset.annotations):
    if name in label:
        idx = i
        break

#target = dataset[_][1]
images, target = next(iter(train_data))
target.requires_grad = True
pred = torch.randn(32, 7, 7, 90, requires_grad=True)
loss = YoloLoss(7, 2, 80)
yolo_loss = loss(pred, target)
print(yolo_loss/32)

#plot_bbox(img, labels)