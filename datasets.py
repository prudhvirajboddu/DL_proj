import torch
import cv2
import numpy as np
import os
import glob as glob

import matplotlib.pyplot as plt
from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE, VALID_DIR
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform
import torchvision.transforms.functional as TF

# the dataset class

"""
This is a custom dataset class that inherits from the PyTorch Dataset class. 
It has three methods: __init__, __getitem__, and __len__.
It returns image and target data which contains the bounding box coordinates and the class labels.
It does transformations on the images and bounding boxes for both training and testing data.
It is used to load the images and annotations from the dataset directory and prepare them for training and validation. 
The __getitem__ method is used to read the image and annotations from the dataset directory and prepare them for training and validation.
The __len__ method is used to return the total number of images in the dataset.
We are going to use these methods to create the training and validation datasets. 
We also can visualize the images with bounding boxes using the visualize_batch function.
"""


class FaceDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.image_paths += glob.glob(f"{self.dir_path}/*.png")
        self.all_images = [image_path.split(
            '/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


# prepare the final datasets and data loaders
train_dataset = FaceDataset(
    TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = FaceDataset(
    VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)


valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")


def visualize_batch(images, targets):
    # Create a figure with subplots
    # fig, axes = plt.subplots(nrows=len(images), ncols=1, figsize=(15,15))
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    axs = axs.ravel()

    # Loop through each image and its targets
    for i, (image, target) in enumerate(zip(images, targets)):
        # Plot the image
        image = TF.to_pil_image(image)
        image = np.array(image)
        axs[i].imshow(image)
        boxes = target['boxes'].numpy()
        labels = [t.item() for t in target['labels'] ]
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
            axs[i].add_patch(rect)
            axs[i].text(x1, y1, CLASSES[labels[j]], fontsize=12, color='red')
    # Show the plot
    plt.show()

