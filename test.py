import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
@st.cache_resource
def load_model_and_labels():
    try:
        model = load_model('model/keras_model_1.h5')  # Adjust path as needed
        with open('model/labels_1.txt', 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        return model, labels
    except Exception as e:
        st.error(f"Error loading model or labels: {e}")
        return None, []

# Preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)  # Add batch dimension

# Predict the class of the image
def predict_image(model, image, class_names):
    data = preprocess_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].encode('utf-8').decode('utf-8')  # Ensure correct encoding
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Function to get nutrition information from a CSV file
def get_nutrition_info(class_name, csv_path):
    try:
        df = pd.read_csv(csv_path)
        nutrition_info = df[df['Food'].str.lower() == class_name.lower()]
        return nutrition_info
    except Exception as e:
        st.error(f"Error loading nutrition data: {e}")
        return None

# Streamlit app layout
def food_app():
    st.title("Food Classification App")

    # Load model and labels
    model, class_names = load_model_and_labels()

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if model is not None:
            prediction_label, probability = predict_image(model, image, class_names)
            st.write(f"Result = {prediction_label[2:]}")  # Strip first two characters if needed
            st.write(f"Confidence = {probability:.2f}")

            # Get nutrition information
            nutrition_info = get_nutrition_info(prediction_label, 'dataset/data.csv')  # Adjust path if needed

            if nutrition_info is not None:
                st.write("Nutrition Table:")
                nutrition_data = {
                    "Nutrition": ["Karbo", "Gula", "Lemak", "Protein", "Serat", "Kolesterol", "Sodium", "Recommended (Yes/No)"],
                    "Value": [
                        round(nutrition_info['Karbo'], 2),
                        round(nutrition_info['Gula'], 2),
                        round(nutrition_info['Lemak'], 2),
                        round(nutrition_info['Protein'], 2),
                        round(nutrition_info['Serat'], 2),
                        round(nutrition_info['Kolesterol'], 2),
                        round(nutrition_info['Sodium'], 2),
                        nutrition_info['Recommended (Yes/No)']
                    ]
                }
                df_nutrition = pd.DataFrame(nutrition_data)
                st.dataframe(df_nutrition.set_index(df_nutrition.columns[0]), use_container_width=True)
            else:
                st.write("No nutrition information found for this food item.")
        else:
            st.error("Model could not be loaded.")

# Run the Streamlit app
if __name__ == "__main__":
    food_app()
