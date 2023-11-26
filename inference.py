import numpy as np
import cv2
import torch
import glob as glob

from model import create_model
from config import NUM_CLASSES, RESIZE_TO
import matplotlib.pyplot as plt

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes = NUM_CLASSES).to(device)

model.load_state_dict(torch.load(
    'outputs/model.pth', map_location=device
))

model.eval()

# directory where all the images are present
DIR_TEST = 'dataset/test'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

test_images = test_images[:8]

# classes: 0 index is reserved for background
CLASSES = [
    'student','Security', 'Staff', 'Facility Worker','Food Service worker'
]

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # resize to the model input size
    image = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        fig, axs = plt.subplots(2, 4, figsize=(15, 10))
        axs = axs.ravel()

        axs[i].imshow(image)
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
            axs[i].add_patch(rect)
            axs[i].text(x1, y1, pred_classes[j], fontsize=12, color='red')

    # save the output image
plt.show()