import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from model import create_model
from config import RESIZE_TO, CLASSES , NUM_CLASSES

# Load the PyTorch face detection model
model = create_model(num_classes = NUM_CLASSES)

model.load_state_dict(torch.load(
    'outputs/model.pth' ))
model.eval()


# Define the function to perform face detection
def detect_faces(image):
    # resize to the model input size
    image = np.reshape(image, (RESIZE_TO, RESIZE_TO, 3))
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    # create a mini-batch as expected by the model
    input_tensor = image.unsqueeze(0)  

    # Run the model inference
    with torch.no_grad():
        output = model(input_tensor)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in output]

    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    # filter out boxes according to `detection_threshold`
    boxes = boxes[scores >= 0.45].astype(np.int32)
    # get all the predicited class names
    pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
    # Get the bounding box coordinates and draw them on the image
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    for j, box in enumerate(boxes):
        x, y, w, h = box
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
        label = pred_classes[j]  # Replace 0 with the appropriate class index
        draw.text((x, y), label, fill="red")  # Adjust the text color and position as needed

    return image_draw

# Define the Gradio interface
iface = gr.Interface(
    fn=detect_faces,
    inputs="image",
    outputs="image",
    title="Face Detection",
    description="Detect faces in an image and draw bounding boxes",
    allow_flagging=False
)

# Run the Gradio interface
iface.launch()
