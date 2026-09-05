import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("model/food101_model.keras")

MAX_SIDE_LEN = 128

st.title("🌭 Hot Dog Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Same preprocessing used during training
    image = image.resize((MAX_SIDE_LEN, MAX_SIDE_LEN))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    # Model returns a logit, so convert it to probability
    logit = model.predict(image, verbose=0)[0][0]
    probability = tf.sigmoid(logit).numpy()

    if probability >= 0.5:
        st.success(f"🌭 Hot Dog! ({probability * 100:.2f}% confidence)")
    else:
        st.error(f"❌ Not a Hot Dog ({(1 - probability) * 100:.2f}% confidence)")