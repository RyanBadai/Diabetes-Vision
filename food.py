import streamlit as st
import requests
from PIL import Image
import io

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

def main():
    st.title("Food Classification App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        prediction = predict_image(img_byte_arr)

        if prediction:
            st.write("Prediction:")
            for pred in prediction["predictions"]:
                st.write(f"{pred['tagName']}: {pred['probability']*100:.2f}%")

if __name__ == "__main__":
    main()
