import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    mri_model = tf.keras.models.load_model("mri_detector.keras")
    tumor_model = tf.keras.models.load_model("brain_tumor_model.keras")
    return mri_model, tumor_model

mri_model, tumor_model = load_models()

# -------------------------------
# Tumor Classes
# -------------------------------
classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# -------------------------------
# Title
# -------------------------------
st.title("🧠 Brain Tumor Detection using Deep Learning")

st.write(
    "Upload a Brain MRI image to detect the tumor type."
)

st.info(
    "This application first checks whether the uploaded image is a Brain MRI. "
    "Only valid Brain MRI images are analyzed for tumor prediction."
)

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose a Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -------------------------------
    # Image Preprocessing
    # -------------------------------
    img = image.resize((128, 128))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # -------------------------------
    # STEP 1 : MRI Detection
    # -------------------------------
    mri_prediction = mri_model.predict(img, verbose=0)

    # Print raw output (for debugging)
    st.write("MRI Model Output:", mri_prediction)

    mri_score = float(mri_prediction[0][0])

    st.write("MRI Score:", round(mri_score, 4))

    # Labels:
    # MRI = 0
    # NonMRI = 1

    if mri_score >= 0.5:
        st.error("❌ Invalid Image! The uploaded image is classified as Non-MRI.")
        st.stop()

    st.success("✅ Brain MRI image detected.")

    # -------------------------------
    # STEP 2 : Tumor Detection
    # -------------------------------
    prediction = tumor_model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    st.success(f"Prediction: **{classes[predicted_class]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    # -------------------------------
    # Prediction Probabilities
    # -------------------------------
    st.subheader("Prediction Probabilities")

    for i, cls in enumerate(classes):
        probability = float(prediction[0][i])

        st.write(f"**{cls}: {probability*100:.2f}%")

        st.progress(probability)