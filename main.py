import io # for handling byte streams (Since HTTP uploads are bytes, not image files)
import numpy as np
from fastapi import FastAPI,File,UploadFile # FastAPI framework to build the REST API
from fastapi.responses import StreamingResponse # sends binary data (image) back to the client
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models,transforms
from PIL import Image,ImageDraw # image loading & drawing bounding boxes.

# Load custom trained model
num_classes = 2  # background + gun
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
path = 'artifacts/models/fasterrcnn.pth'
model.load_state_dict(torch.load(path, map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # inference mode 


# Convert PIL images → PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

app = FastAPI() # Create the FastAPI application object

def predict_and_draw(image : Image.Image):

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(): # Run inference without storing gradients
        predictions = model(img_tensor)

    # Extract bounding boxes, class labels, and confidence scores
    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    img_rgb = image.convert("RGB") # Ensure the image is in RGB

    draw = ImageDraw.Draw(img_rgb) # prepare a drawing canvas

    # Loop through predictions, only draw bounding boxes with confidence > 70%
    for box,score in zip(boxes,scores):
        if score>0.7:
            x_min,y_min,x_max,y_max = box
            draw.rectangle([x_min,y_min,x_max,y_max] , outline="red" , width=2)
    
    return img_rgb

# Root endpoint → sanity check if server is running
@app.get("/")
def read_root():
    return {"message" : "Welcome to the Guns Object Detection API"}


@app.post("/predict/")
async def predict(file:UploadFile=File(...)): # Accept file upload via HTTP POST; here file is the key that will give to postman

    image_data = await file.read() # Read raw bytes 
    image = Image.open(io.BytesIO(image_data)) # convert back into PIL Image

    output_image = predict_and_draw(image)

    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr , format='PNG') # Save processed image (with boxes) into memory (not disk)
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr , media_type="image/png") # Send processed image back as PNG response

