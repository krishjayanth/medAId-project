import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the models
binary_model = tf.keras.models.load_model('binary_classification_model(1).keras')
multiclass_model = tf.keras.models.load_model('multiclass_classification_model_v5.keras')

# Define function to classify a single image
def classify_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Stage 1: Binary classification
        binary_prediction = binary_model.predict(img_array)
        if binary_prediction > 0.5:
            return "Normal"
        else:
            # Stage 2: Multiclass classification
            multiclass_prediction = multiclass_model.predict(img_array)
            cancer_types = ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma']
            predicted_cancer_type = cancer_types[np.argmax(multiclass_prediction)]
            return predicted_cancer_type
    except Exception as e:
        print(f"Error: {e}")

# Example usage
image_path = '/working_adenocarcinoma.png'
result = classify_image(image_path)
print(f'The image is classified as: {result}')
