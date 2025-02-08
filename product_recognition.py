import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from flask import Flask, render_template, request
import requests
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained ResNet-50 model with updated weights
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to("cpu")
model.eval()

# Correct way to load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS = requests.get(LABELS_URL).text.split("\n")

# Image transformation for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

'''
# Function to recognize product from an image (accepts image file object)
def recognize_product_from_image(image):
    """Recognize product from image."""
    image = Image.open(image).convert("RGB")  # Ensure image is in RGB mode
    image = transform(image).unsqueeze(0)  # Transform image

    with torch.no_grad():
        outputs = model(image)  # Make prediction

    _, predicted_idx = torch.max(outputs, 1)
    product_name = LABELS[predicted_idx.item()]  # Get predicted class name

    return product_name
'''
# Function to recognize product from an image (accepts image file object)
def recognize_product_from_image(image):
    """Recognize product from image with memory optimization."""
    image = Image.open(image).convert("RGB")  # Ensure image is in RGB mode
    image = transform(image).unsqueeze(0)  # Transform image

    # Move model and image to CPU (to reduce memory usage)
    model.to("cpu")
    image = image.to("cpu")

    # Disable gradient calculations to save memory
    with torch.no_grad():
        outputs = model(image)  # Make prediction

    _, predicted_idx = torch.max(outputs, 1)
    product_name = LABELS[predicted_idx.item()]  # Get predicted class name

    return product_name

# Flask route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for searching (both text and image-based)
@app.route('/search', methods=['POST'])
def search():
    # Check if the user has entered a product name (text-based search)
    product_name = request.form.get('product_name', '')
    image = request.files.get('product_image')

    if product_name:  # Text-based search
        # Handle text-based search (not implemented here)
        pass

    elif image:  # Image-based search
        if image.filename == '':
            return "No selected file"

        # Recognize product using DL model
        recognised_product_name = recognize_product_from_image(image)
        print("Recognized Product:", recognised_product_name)

        # Perform product search (replace this with your actual search logic)
        results = []  # Replace with actual API search logic
        print("Search Results for image:", results)

        # Return the results to the search results page
        return render_template('searchresults.html', results=results, query=recognised_product_name)

    else:
        return "No product name or image provided"

if __name__ == '__main__':
    app.run(debug=True)
