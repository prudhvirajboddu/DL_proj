import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from model import create_model
from config import RESIZE_TO, CLASSES , NUM_CLASSES

# Load the PyTorch face detection model
model = create_model(num_classes = NUM_CLASSES)

model.load_state_dict(torch.load('outputs/mmodel_11.pth' ))
model.eval()


# Define the function to perform face detection
def detect_faces(image):

    target_size = (RESIZE_TO, RESIZE_TO)
    
    image = cv2.resize(image, target_size)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = image / 255.0

    # image_draw = image.copy()

    # resize to the model input size and add a batch dimension
    image = torch.from_numpy(image).float().unsqueeze(0).permute(0,3,1,2)  

    # Run the model inference
    with torch.no_grad():
        output = model(image)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in output]

    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()

    print(scores[:])

    # filter out boxes according to `detection_threshold`
    boxes = boxes[scores >= 0.5].astype(np.int32)

    # get all the predicited class names
    pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

    # Get the bounding box coordinates and draw them on the image

    image_draw = image.clone().squeeze(0).permute(1, 2, 0).numpy()

    # Convert the NumPy array to a PIL Image
    image_draw = (image_draw * 255).astype(np.uint8)
    image_draw = Image.fromarray(image_draw)
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
