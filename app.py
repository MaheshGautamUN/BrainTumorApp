import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----------------------
# 1️⃣ Load your model
# ----------------------
MODEL_PATH = r"C:\Users\mahes\OneDrive - United Nations\Data Science course Naresh\3 Data Science\Aritificial Intelegence - AI\November\9 Nov\Brain-Tumor-Classification-DataSet-master\brain_tumor_classifier_v1.keras"

@st.cache_resource
def load_model(path):
    return keras.models.load_model(path)

model = load_model(MODEL_PATH)

# ----------------------
# 2️⃣ Labels (same as training)
# ----------------------
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# ----------------------
# 3️⃣ Streamlit UI
# ----------------------
st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image to classify the brain tumor type.")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

# ----------------------
# 4️⃣ Process uploaded image
# ----------------------
if uploaded_file is not None:
    # Open and display the image
    img_display = Image.open(uploaded_file)
    st.image(img_display, caption="Uploaded MRI Image", use_column_width=True)

    # Convert to RGB if not already (some MRI images may be grayscale)
    img_display = img_display.convert("RGB")

    # Preprocess the image (resize, array, normalize)
    img = image.load_img(uploaded_file, target_size=(150,150))  # resize same as training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize like training

    # ----------------------
    # 5️⃣ Predict fresh for each image
    # ----------------------
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display results
    st.markdown(f"### 🧩 Predicted Tumor Type: **{predicted_class}**")
    st.markdown(f"### 🔹 Confidence: **{confidence:.2f}**")
    st.write("### Note: This is a demo app. For medical diagnosis, please consult a healthcare professional.")  