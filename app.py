import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model and processor (from Hugging Face)
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_models()

# Function to generate caption and dummy emotion
def generate_caption(image, style="old"):
    raw_image = image.convert('RGB')

    # Style-based prompt
    if style == "old":
        prompt = "An old-fashioned poetic description of the image:"
    else:
        prompt = "A modern and casual description of the image:"

    # Pass prompt + image to processor
    inputs = processor(raw_image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50, num_beams=5)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Dummy emotion (you can integrate your emotion detection model here)
    emotion = "Nostalgic" if style == "old" else "Peaceful"

    return caption, emotion

# Streamlit UI
st.set_page_config(page_title="🖼️ Image Captioning with Emotion", layout="centered")
st.title("🖼️ Image Captioning with Emotion Detection")
st.write("Upload an image and click **Generate** to get captions in two styles along with detected emotions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Only show button when image is uploaded
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("✨ Generate Captions & Emotions"):
        old_caption, old_emotion = generate_caption(image, style="old")
        new_caption, new_emotion = generate_caption(image, style="modern")

        st.markdown("### Older ")
        st.markdown(f"**Emotion:** {old_emotion}")
        st.markdown(f"**Caption:** {old_caption}")

        st.markdown("###  Modern ")
        st.markdown(f"**Emotion:** {new_emotion}")
        st.markdown(f"**Caption:** {new_caption}")

