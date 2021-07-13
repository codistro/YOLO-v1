from bs4 import BeautifulSoup
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import sys
images_root = sys.argv[1]
labels_root = sys.argv[2]
labels_dest = sys.argv[3]

try:
    os.makedirs(labels_dest)
except FileExistsError:
    pass

labels_file = sys.argv[4]
f = open(labels_file, 'r').readlines()
label_to_idx = {}
for _ in f:
    label, idx = _.split()
    label_to_idx[label.lower()] = idx

def plot_bbox(img, labels):
    fig, ax = plt.subplots()
    ax.imshow(img)

    for _ in labels:
        cls, x, y, w, h = _.split()
        x, y, w, h = float(x), float(y), float(w), float(h)
        print(x, y, w, h)
        ax.add_patch(patches.Rectangle(
            (x, y),w-x,h-y,
            linewidth=1,
            edgecolor='b',
            facecolor='none'
        ))

    plt.show()

def convert_coordinates(img_root, labels_root):

    for label in tqdm(os.listdir(labels_root)):
        labels_file = os.path.join(labels_root, label)
        img_file = os.path.join(img_root, label.split('.')[0]+'.jpg')
        img = Image.open(img_file)
        img_width = img.width
        img_height = img.height
        with open(labels_file) as f:
            data = f.read()
        xml_data = BeautifulSoup(data, "xml")
        objects = xml_data.find_all('object')
        coord = []
        for object in objects:
            name = object.find('name').contents[0]
            bbox = object.find('bndbox')
            x1, y1, x2, y2 = bbox.find('xmin').contents[0], bbox.find('ymin').contents[0], bbox.find('xmax').contents[0], bbox.find('ymax').contents[0]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            w = x2 - x1
            h = y2 - y1

            x = x1 + w / 2
            y = y1 + h / 2

            x /= img_width
            y /= img_height
            w /= img_width
            h /= img_height
            label_idx = label_to_idx[str(name).strip().lower()]
            coord.append(' '.join([str(label_idx), str(x), str(y), str(w), str(h)]))

        label = label.split('.')[0] + '.txt'
        with open(os.path.join(labels_dest, label), 'w') as f:
            coord = '\n'.join([x for x in coord])
            f.write(coord)



convert_coordinates(images_root, labels_root)

