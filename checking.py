import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Resizing, Rescaling

# Load your trained model
model = load_model('./models/fruits.h5')  # Update with your model path

# Fruit class names corresponding to model predictions
class_names = [
    'freshapples',
    'freshbanana',
    'freshoranges',
    'rottenapples',
    'rottenbanana',
    'rottenoranges'
]  # Update this list with your actual class names

# Define image size
IMAGE_SIZE = 300

# Create a preprocessing model using Keras layers
resize_and_rescale = Sequential([
    Resizing(IMAGE_SIZE, IMAGE_SIZE),   # Resize images
    Rescaling(1. / 255)                 # Rescale pixel values
])

# Function to preprocess the image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = resize_and_rescale(tf.convert_to_tensor(img))  # Resize and rescale using Keras layers
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict fruit freshness and name
def predict_freshness(image):
    processed_image = preprocess_image(image)
    print(f'Processed image shape: {processed_image.shape}')  # Debug: print shape
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_name = class_names[predicted_index]
    freshness_score = prediction[0][predicted_index]
    return predicted_name, freshness_score

# Capture real-time video
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get prediction
    fruit_name, freshness_score = predict_freshness(frame)
    
    # Display the resulting frame with fruit name and freshness score
    cv2.putText(frame, f'Fruit: {fruit_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Freshness: {freshness_score:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Fruit Freshness Checker', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
