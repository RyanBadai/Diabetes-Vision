import io
import pandas as pd
import numpy as np
import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image
import tensorflow as tf

def food_app():
    st.title("Diabetes Vision")

    st.write(f'<div style="text-align: justify;">Diabetes Vision provides a comprehensive diabetes risk analysis tool. We direct users to upload photos of the food they consume, and our site will detect the carbohydrate and sugar content and other nutritional values of the uploaded food photos. This information becomes our reference to recommend whether the food is suitable or not for the user based on the risk analysis that has been done.</div>', unsafe_allow_html=True)

    def calculate_score(gender, weight, height, age, waist_circumference, family_history, lifestyle):
        # BMI calculation
        bmi = weight / (height / 100) ** 2

        # BMI score
        if bmi < 18.5:
            bmi_score = 0
        elif 18.5 <= bmi <= 22.9:
            bmi_score = 0
        elif 23 <= bmi <= 24.9:
            bmi_score = 1
        elif 25 <= bmi <= 29.9:
            bmi_score = 2
        else:
            bmi_score = 3

        # Age score
        if age < 35:
            age_score = 0
        elif 35 <= age <= 44:
            age_score = 0
        elif 45 <= age <= 54:
            age_score = 2
        elif 55 <= age <= 64:
            age_score = 3
        else:
            age_score = 4

        # Waist circumference score
        if gender == "Laki-laki":
            if waist_circumference < 80:
                waist_score = 0
            elif 80 <= waist_circumference <= 88:
                waist_score = 0
            elif 89 <= waist_circumference <= 93:
                waist_score = 3
            elif 94 <= waist_circumference <= 102:
                waist_score = 3
            else:
                waist_score = 4
        else:
            if waist_circumference < 80:
                waist_score = 0
            elif 80 <= waist_circumference <= 88:
                waist_score = 3
            elif 89 <= waist_circumference <= 93:
                waist_score = 4
            elif 94 <= waist_circumference <= 102:
                waist_score = 4
            else:
                waist_score = 4

        # Family history score
        family_history_score = 0
        if family_history['parent_sibling_child'] == 'Yes':
            family_history_score += 5
        if family_history['grandparent_uncle_aunt_cousin'] == 'Yes':
            family_history_score += 3
        if family_history['high_blood_sugar'] == 'Yes':
            family_history_score += 5
        if family_history['hypertension_meds'] == 'Yes':
            family_history_score += 2

        # Lifestyle score
        lifestyle_score = 0
        if lifestyle['exercise'] == 'No':
            lifestyle_score += 2
        if lifestyle['fruits_veggies'] == 'No':
            lifestyle_score += 1

        # Total score
        total_score = bmi_score + age_score + waist_score + family_history_score + lifestyle_score

        return total_score

    def create_gauge(score):
        if score < 7:
            color = "green"
            risk_category = "Low Risk"
        elif score <= 11:
            color = "yellow"
            risk_category = "Slightly High Risk"
        elif score <= 14:
            color = "orange"
            risk_category = "Moderate Risk"
        elif score <= 20:
            color = "red"
            risk_category = "High Risk"
        else:
            color = "darkred"
            risk_category = "Very High Risk"

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': "Blood Sugar Score"},
            gauge = {
                'axis': {'range': [0, 20]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 7], 'color': 'lightgreen'},
                    {'range': [7, 11], 'color': 'yellow'},
                    {'range': [11, 14], 'color': 'orange'},
                    {'range': [14, 20], 'color': 'red'},
                    {'range': [20, 30], 'color': 'darkred'}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score}}))

        fig.add_annotation(
            x=0.5,
            y=0.4,
            text=risk_category,
            showarrow=False,
            font=dict(size=20)
        )

        return fig

    # Load the model
    models = [
        tf.keras.models.load_model('model/keras_model_1.h5'),
        tf.keras.models.load_model('model/keras_model_2.h5'),
        tf.keras.models.load_model('model/keras_model_3.h5'),
        tf.keras.models.load_model('model/keras_model_4.h5')]

    # Load labels from labels.txt
    labels_list = []
    for i in range(1, 5):
        with open(f'model/labels_{i}.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            labels_list.append(labels)

    def preprocess_image(image, target_size):
        # Resize image to target size
        image = image.resize(target_size)
        # Convert image to array
        image_array = np.array(image)
        # Scale the pixel values (normalizing if necessary)
        image_array = image_array / 255.0
        # Expand dimensions to match model input shape (batch size)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict_image(image):
        processed_image = preprocess_image(image, target_size=(224, 224))  # Assuming 224x224 input for the models
        
        predictions_combined = []
        for model in models:
            predictions_combined.append(model.predict(processed_image))

        top_predictions = []
        for i, predictions in enumerate(predictions_combined):
            top_prediction_index = np.argmax(predictions[0])
            predicted_label = labels_list[i][top_prediction_index]
            probability = predictions[0][top_prediction_index]
            top_predictions.append((predicted_label, probability))
        
        # Find the model with the highest probability
        best_prediction = max(top_predictions, key=lambda x: x[1])
        return best_prediction

    def get_nutrition_info(food_name, csv_file):
        df = pd.read_csv(csv_file)
        food_info = df[df['Nama Makanan'].str.lower() == food_name.lower()]
        if not food_info.empty:
            return food_info.iloc[0]
        else:
            return None

    st.write(" ")
    tabs = st.tabs(["Diabetes Risk Analysis", "Food Classification  "])

    with tabs[0]:
        # st.write("## Perhitungan Skor Gula Darah")

        # Input fields
        gender = st.selectbox("Gender / Jenis Kelamin", ["Male", "Female"], index=None)
        weight = st.number_input("Weight / Berat Badan (kg)", min_value=0)
        height = st.number_input("Height / Tinggi Badan (cm)", min_value=0)
        age = st.number_input("Age / Usia (Years / Tahun)", min_value=0)
        waist_circumference = st.number_input("Waistline / Lingkar Pinggang (cm)", min_value=0)

        family_history = {
            'parent_sibling_child': st.radio("Have your parents, siblings, or children ever been diagnosed with diabetes / Apakah orang tua, saudara kandung, atau anak kandung pernah didiagnosis Diabetes?", ["Yes", "No"], index=None),
            'grandparent_uncle_aunt_cousin': st.radio("Have your grandparents, uncles, aunts, or first cousins ever been diagnosed with diabetes / Apakah kakek, bibi, paman, atau sepupu pertama pernah didiagnosis Diabetes ?", ["Yes", "No"], index=None),
            'high_blood_sugar': st.radio("Have you ever had high blood sugar levels / Apakah kamu pernah memiliki kadar gula darah yang tinggi?", ["Yes", "No"], index=None),
            'hypertension_meds': st.radio("Have you ever taken medication for high blood pressure / Apakah kamu pernah mengkonsumsi obat darah tinggi?", ["Yes", "No"], index=None)
        }

        lifestyle = {
            'exercise': st.radio("Do you exercise for more than 30 minutes every day / Apakah kamu berolahraga selama lebih dari 30 menit setiap hari?", ["Yes", "No"], index=None),
            'fruits_veggies': st.radio("Do you consume fruits and vegetables every day / Apakah Anda Mengonsumsi Buah dan Sayuran setiap hari?", ["Yes", "No"], index=None)
        }

        if st.button("Result"):
            score = calculate_score(gender, weight, height, age, waist_circumference, family_history, lifestyle)
        
            if score < 7:
                st.success("Low Risk of Diabetes. An estimated 1 in 100 people will develop diabetes")
            elif score <= 11:
                st.success("Slightly High Risk of Diabetes. It is estimated that 1 in 25 people will have diabetes")
            elif score <= 14:
                st.success("Moderate Risk of Diabetes. It is estimated that 1 in 6 people will have Diabetes")
            elif score <= 20:
                st.success("High Risk of Diabetes. It is estimated that 1 in 3 people will have Diabetes")
            else:
                st.success("Very High Risk of Diabetes. It is estimated that 1 in 2 people will have Diabetes")
            
            fig = create_gauge(score)
            st.plotly_chart(fig)

    with tabs[1]:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            prediction_label, probability = predict_image(image)

            if prediction_label:
                st.write(f"Result = {prediction_label}")
                st.write(f"Confidence = {probability:.2f}")

                # Get nutrition information
                nutrition_info = get_nutrition_info(prediction_label, 'dataset/data.csv')
                
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


