import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model #type:ignore

# Page config
st.set_page_config(page_title="Image Classifier", layout="centered")

# Title
st.title("🧠 Image Classifier Using CNN")
st.write("Upload an image and get prediction instantly")

# Load model
@st.cache_resource
def load_cnn_model():
    return load_model("models/cifar_cnn_model.h5")

model = load_cnn_model()

# Class labels
class_names = ['airplane','car','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ✅ EVERYTHING must be inside this block
if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Show image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Get top 3 predictions
    top_indices = np.argsort(prediction[0])[-3:][::-1]

    # Show result
    st.markdown("### 🔍 Prediction Result")
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}**")

    # Explanation
    st.markdown("### 🧠 Explanation")

    if confidence > 0.80:
        st.write("The model is highly confident about this prediction.")
    elif confidence > 0.50:
        st.write("The model is moderately confident. There may be similar classes.")
    else:
        st.write("The model is not very confident. The image may be unclear or belong to another class.")

    # Top 3 predictions
    st.markdown("### 📊 Top 3 Predictions")
    for i in top_indices:
        st.write(f"{class_names[i]} → {prediction[0][i]:.2f}")