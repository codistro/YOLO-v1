import os
from PIL import Image, ImageFile
from torchvision import transforms

from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.ToTensor()

image_path = ['/content/val2014']
label_path = ['/content/labels/val2014']


def remove_gray_images(img_path, label_path):
    count = 0
    for image in tqdm(os.listdir(img_path)):

        img_file = os.path.join(img_path, image)
        pil_img = Image.open(img_file)
        img_file_name = image.split('.')[0]
        label_file = os.path.join(label_path, img_file_name + '.txt')
        if pil_img.mode != 'RGB':
            try:
                pil_img.close()
                os.remove(img_file)
                os.remove(label_file)
            except FileNotFoundError:
                print('File Not Found')
            count += 1

    print("Total Gray Images: {}".format(count))


for img_path, labels in zip(image_path, label_path):
    remove_gray_images(img_path, labels)
