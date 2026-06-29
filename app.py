import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# Load Trained Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")

model = load_model()

# Class Labels
classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# Title
st.title("🧠 Brain Tumor Detection")
st.write("Upload a Brain MRI image to predict the tumor type.")

# Upload Image
uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display results
    st.success(f"Prediction: {classes[predicted_class]}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Show probabilities
    st.subheader("Prediction Probabilities")

    for i, cls in enumerate(classes):
        st.write(f"**{cls}: {prediction[0][i] * 100:.2f}%**")
        st.progress(float(prediction[0][i]))