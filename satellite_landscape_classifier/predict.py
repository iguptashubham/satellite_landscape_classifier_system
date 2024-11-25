from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model(r'C:\Users\gupta\OneDrive\Desktop\Projects\satellite_land_classification\satellite_landscape_classification\models\SLCS_model_20241124-151414\SLCS.h5')

# Function to preprocess images
def processimage(img_path):
    img = image.load_img(img_path, target_size=(255, 255))  # Ensure target size matches the model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image data to the range [0, 1]
    return img_array

def make_prediction(processed_img):

# Make predictions
    model = load_model(r'C:\Users\gupta\OneDrive\Desktop\Projects\satellite_land_classification\satellite_landscape_classification\models\SLCS_model_20241124-151414\SLCS.h5')
    predictions = model.predict(processed_img)
    class_names = ['cloudy', 'desert', 'green_area', 'water']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name
