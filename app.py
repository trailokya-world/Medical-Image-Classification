import streamlit as st
import numpy as np
from PIL import Image
# from tensorflow import keras
import keras


st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMAGE_SIZE  = (224, 224)


@st.cache_resource
def load_model():
    return keras.models.load_model("model_keras.keras") 

model = load_model()

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI scan to classify the tumor type.")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)

    if st.button("Analyze"):
        # Preprocess
        img = image.resize(IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        # prediction = model(img_array)
        preds    = list(model.predict(img_array))[0]
        pred_idx = np.argmax(preds)
        pred_cls = CLASS_NAMES[pred_idx]
        conf     = preds[pred_idx] * 100

        # Results
        st.success(f"**Prediction: {pred_cls}** ({conf:.1f}% confidence)")
        st.write("### All Class Probabilities")

        for cls, prob in zip(CLASS_NAMES, preds):
            st.write(f"{cls}")
            st.progress(float(prob), text=f"{prob*100:.1f}%")