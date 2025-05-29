import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
@st.cache_resource
def load_smartleaf_model():
    return load_model("smartLeaf_mobilenetV2.h5")

model = load_smartleaf_model()

# Set class labels
class_labels = [
    'Corn___Common_Rust',
    'Corn___Gray_Leaf_Spot',
    'Corn___Healthy',
    'Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight',
    'Potato___Healthy',
    'Potato___Late_Blight',
    'Rice___Brown_Spot',
    'Rice___Healthy',
    'Rice___Leaf_Blast',
    'Rice___Neck_Blast',
    'Wheat___Brown_Rust',
    'Wheat___Healthy',
    'Wheat___Yellow_Rust'
]

# Title
st.title("🌿 SmartLeaf - Crop Disease Detector")

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((224, 224))  # Change if your model uses different input size
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)

    # Debug output
    st.write("Raw prediction output:", prediction)
    st.write("Prediction shape:", prediction.shape)

    # Handle classification
    if prediction.shape[1] == len(class_labels):
        predicted_class = class_labels[np.argmax(prediction)]
        st.success(f"Predicted: **{predicted_class}**")
        st.write("Confidence:", np.round(np.max(prediction) * 100, 2), "%")
    else:
        st.error("Mismatch between model output size and class_labels. Check class_labels.")
