import streamlit as st
from PIL import Image
import numpy as np

# Dummy placeholder functions – replace with your actual implementations
def generate_caption_and_emotion(image, style='old'):
    if style == 'old':
        return "A classical artwork showing a serene village scene.", "Calm"
    else:
        return "A peaceful countryside landscape under the sunset.", "Peaceful"

# Streamlit App
st.set_page_config(page_title="Image Captioning & Emotion Detection", layout="centered")

st.title("🖼️ Image Captioning with Emotion Detection")
st.write("Upload an image to generate **older** and **modern** captions with associated **emotions**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("🔍 Generating captions and emotions...")

    # Run your models (replace with actual model inference)
    old_caption, old_emotion = generate_caption_and_emotion(image, style='old')
    modern_caption, modern_emotion = generate_caption_and_emotion(image, style='modern')

    # Display results
    st.subheader("🕰️ Older Style")
    st.markdown(f"**Caption:** {old_caption}")
    st.markdown(f"**Emotion:** {old_emotion}")

    st.subheader("🧠 Modern Style")
    st.markdown(f"**Caption:** {modern_caption}")
    st.markdown(f"**Emotion:** {modern_emotion}")
