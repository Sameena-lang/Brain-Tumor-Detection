import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Class Names
classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# Title
st.title("🔥 SAMEENA'S BRAIN TUMOR DETECTION APP 🔥")

st.write("Upload a Brain MRI image to predict the tumor type.")

# Upload Image
uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess Image
    img = image.resize((128, 128))
    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {classes[predicted_class]}")

    st.info(f"Confidence: {confidence:.2f}%")