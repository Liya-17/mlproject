# classifier/load_model.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained model
model = MobileNetV2(weights="imagenet")

def classify_image(img_path, top_n=3, confidence_threshold=0.5):
    """
    Classifies an image using the MobileNetV2 model and returns the top predictions 
    with confidence scores.

    Parameters:
    - img_path: Path to the image file.
    - top_n: Number of top predictions to return.
    - confidence_threshold: Minimum confidence score required to return a prediction.

    Returns:
    - List of tuples with (label, description, confidence) for each prediction above the threshold.
      If no predictions meet the threshold, returns a default message.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=top_n)[0]

    # Filter predictions based on confidence threshold
    filtered_preds = [
        (label, description, score) for label, description, score in decoded_preds 
        if score >= confidence_threshold
    ]
    
    # Return filtered predictions or a default message if none meet the threshold
    return filtered_preds if filtered_preds else [("N/A", "No highly confident prediction found", 0.0)]
