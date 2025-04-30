import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import json
import os

def load_model_and_indices():
    # Load the trained model
    model = tf.keras.models.load_model('dog_breed_classifier.h5')
    
    # Load class indices
    with open('class_indices.json', 'r') as f:
        idx_to_class = json.load(f)
    
    # Print class indices for debugging
    print("\nAvailable class indices:")
    for idx, name in idx_to_class.items():
        print(f"Index {idx}: {name}")
    
    return model, idx_to_class

def preprocess_image(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_breed(model, img_array, idx_to_class):
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Debug information
    print(f"\nRaw prediction indices: {predictions[0]}")
    print(f"Predicted class index: {predicted_class}")
    
    # Get the breed name
    breed_name = idx_to_class[str(predicted_class)]
    
    return breed_name, confidence

def main():
    # Load model and class indices
    model, idx_to_class = load_model_and_indices()
    
    # Example usage
    while True:
        img_path = input("\nEnter the path to the dog image (or 'q' to quit): ")
        if img_path.lower() == 'q':
            break
            
        if not os.path.exists(img_path):
            print("File not found. Please try again.")
            continue
            
        try:
            # Preprocess the image
            img_array = preprocess_image(img_path)
            
            # Make prediction
            breed_name, confidence = predict_breed(model, img_array, idx_to_class)
            
            # Display results
            print(f"\nPredicted Breed: {breed_name}")
            print(f"Confidence: {confidence:.2%}\n")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print("Please ensure you're using the correct model and class indices files.")

if __name__ == "__main__":
    main() 