import io
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image

st.title("Diabetes Vision")

st.write(f'<div style="text-align: justify;">Diabetes Vision menyediakan tools analisis risiko diabetes yang ​komprehensif. Kami mengarahkan pengguna untuk mengunggah ​foto makanan yang dikonsumsi, dan situs kami akan mendeteksi ​kandungan karbohidrat dan gula serta informasi lain dari foto ​makanan yang diunggah. Sehingga kami dapat merekomendasikan ​apakah makanan tersebut cocok untuk pengguna berdasarkan ​analisis risiko yang telah dilakukan.</div>', unsafe_allow_html=True)

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
    if family_history['parent_sibling_child'] == 'Ya':
        family_history_score += 5
    if family_history['grandparent_uncle_aunt_cousin'] == 'Ya':
        family_history_score += 3
    if family_history['high_blood_sugar'] == 'Ya':
        family_history_score += 5
    if family_history['hypertension_meds'] == 'Ya':
        family_history_score += 2

    # Lifestyle score
    lifestyle_score = 0
    if lifestyle['exercise'] == 'Tidak':
        lifestyle_score += 2
    if lifestyle['fruits_veggies'] == 'Tidak':
        lifestyle_score += 1

    # Total score
    total_score = bmi_score + age_score + waist_score + family_history_score + lifestyle_score

    return total_score

def create_gauge(score):
    # Menentukan warna dan pembagian risiko
    if score < 7:
        color = "green"
        risk_category = "Risiko Rendah"
    elif score <= 11:
        color = "yellow"
        risk_category = "Risiko Sedikit Tinggi"
    elif score <= 14:
        color = "orange"
        risk_category = "Risiko Sedang/Moderat"
    elif score <= 20:
        color = "red"
        risk_category = "Risiko Tinggi"
    else:
        color = "darkred"
        risk_category = "Risiko Sangat Tinggi"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Skor Gula Darah"},
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

# Set your Custom Vision Prediction API details
prediction_key = "9f3b60cb7bd94a52a521b1687039efc5"
endpoint = "https://customvisionwebcastss-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/a2754bc9-cf96-43b6-ba20-3550965f05a8/classify/iterations/Iteration3/image"

def predict_image(image):
    headers = {
        "Prediction-Key": prediction_key,
        "Content-Type": "application/octet-stream"
    }

    response = requests.post(endpoint, headers=headers, data=image)

    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error: " + str(response.status_code))
        return None

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
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=None)
    weight = st.number_input("Berat Badan (kg)", min_value=1)
    height = st.number_input("Tinggi Badan (cm)", min_value=1)
    age = st.number_input("Usia (tahun)", min_value=1)
    waist_circumference = st.number_input("Lingkar Pinggang (cm)", min_value=1)

    family_history = {
        'parent_sibling_child': st.radio("Riwayat Keluarga (Orang Tua/Saudara/Keturunan)", ["Ya", "Tidak"], index=None),
        'grandparent_uncle_aunt_cousin': st.radio("Riwayat Keluarga (Kakek/Nenek/Paman/Bibi/Sepupu)", ["Ya", "Tidak"], index=None),
        'high_blood_sugar': st.radio("Riwayat Gula Darah Tinggi", ["Ya", "Tidak"], index=None),
        'hypertension_meds': st.radio("Mengonsumsi Obat Hipertensi", ["Ya", "Tidak"], index=None)
    }

    lifestyle = {
        'exercise': st.radio("Apakah Anda Berolahraga?", ["Ya", "Tidak"], index=None),
        'fruits_veggies': st.radio("Apakah Anda Mengonsumsi Buah dan Sayuran?", ["Ya", "Tidak"], index=None)
    }

    if st.button("Hasil"):
        score = calculate_score(gender, weight, height, age, waist_circumference, family_history, lifestyle)
    
        if score < 7:
            st.success("Risiko Rendah Diabetes. Diperkirakan 1 dari 100 orang akan mengidap Diabetes")
        elif score <= 11:
            st.success("Risiko Sedikit Tinggi Diabetes. Diperkirakan 1 dari 25 orang akan mengidap Diabetes")
        elif score <= 14:
            st.success("Risiko Sedang/Moderat Diabetes. Diperkirakan 1 dari 6 orang akan mengidap Diabetes")
        elif score <= 20:
            st.success("Risiko Tinggi Diabetes. Diperkirakan 1 dari 3 orang akan mengidap Diabetes")
        else:
            st.success("Risiko Sangat Tinggi Diabetes. Diperkirakan 1 dari 2 orang akan mengidap Diabetes")
        
        fig = create_gauge(score)
        st.plotly_chart(fig)

with tabs[1]:
    # st.write("## Klasifikasi Makanan")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        prediction = predict_image(img_byte_arr)

        if prediction:
            # Get the top prediction
            top_prediction = max(prediction["predictions"], key=lambda x: x['probability'])
            food_name = top_prediction['tagName'].capitalize()
            st.write(f"Hasil = {food_name}")
        
            # Get nutrition information
            nutrition_info = get_nutrition_info(top_prediction['tagName'], 'data.csv')
            
            if nutrition_info is not None:
                st.write("Tabel Nutrisi:")
                nutrition_data = {
                    "Nutrisi": ["Karbo", "Gula", "Lemak", "Protein", "Serat", "Kolesterol", "Sodium", "Direkomendasikan (Ya/Tidak)"],
                    "Nilai": [
                        round(nutrition_info['Karbo'], 2),
                        round(nutrition_info['Gula'], 2),
                        round(nutrition_info['Lemak'], 2),
                        round(nutrition_info['Protein'], 2),
                        round(nutrition_info['Serat'], 2),
                        round(nutrition_info['Kolesterol'], 2),
                        round(nutrition_info['Sodium'], 2),
                        nutrition_info['Direkomendasikan (Ya/Tidak)']
                    ]
                }
                df_nutrition = pd.DataFrame(nutrition_data)
                st.dataframe(df_nutrition.set_index(df_nutrition.columns[0]), use_container_width=True)
            else:
                st.write("No nutrition information found for this food item.")

