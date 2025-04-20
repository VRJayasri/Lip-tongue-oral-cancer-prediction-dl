import os
import io
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained ResNet model
# Define a compatible loss function
custom_loss = BinaryCrossentropy(reduction="sum_over_batch_size")

# Load the model and override the loss function
model = load_model("Resnet.h5", compile=False)  # Load without compiling
model.compile(optimizer="adam", loss=custom_loss, metrics=["accuracy"])

# Function to load and preprocess a single image for ResNet
def load_and_preprocess_image(file, target_size=(240, 240)):
    """Load and preprocess an image."""
    image = load_img(file, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess the image as per ResNet requirements
    return image

# Function to make a prediction using the ResNet model
def predict_cancer(image):
    """Predict if an image is cancerous or non-cancerous."""
    # Make a prediction
    prediction = model.predict(image)
    
    # Convert prediction to label
    label = "cancer" if prediction >= 0.5 else "non-cancer"
    confidence = prediction[0][0] if label == "cancer" else 1 - prediction[0][0]
    
    return label, float(confidence)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a valid image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if the file is actually an image
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400
    
    if file:
        try:
            # Load and preprocess the uploaded image
            image_stream = io.BytesIO(file.read())
            image = load_and_preprocess_image(image_stream)
            
            # Make prediction
            label, confidence = predict_cancer(image)
            
            # Redirect to result page with prediction data
            return render_template('result.html', label=label, confidence=confidence)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Unknown error occurred'}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
