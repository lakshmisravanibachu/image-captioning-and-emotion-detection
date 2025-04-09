import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import torch
import random

# Load models and tokenizer
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    return blip_processor, blip_model, t5_tokenizer, t5_model

blip_processor, blip_model, t5_tokenizer, t5_model = load_models()

# Emotion mappings
older_emotions = ["nostalgia", "sadness", "mystery", "danger"]
modern_emotions = ["joy", "excitement", "adventure", "humor"]

# Function to transform caption
def transform_caption(text, style, emotion):
    if style == "older":
        if emotion == "nostalgia":
            prompt = f"Rewrite this in an old-fashioned, sentimental style: {text}"
        elif emotion == "sadness":
            prompt = f"Rewrite this as a melancholic and deep story: {text}"
        elif emotion == "mystery":
            prompt = f"Rewrite this in a suspenseful and enigmatic tone: {text}"
        elif emotion == "danger":
            prompt = f"Rewrite this in a thrilling and perilous manner: {text}"
    else:
        if emotion == "joy":
            prompt = f"Make this fun, cheerful, and lighthearted: {text}"
        elif emotion == "excitement":
            prompt = f"Make this energetic and thrilling: {text}"
        elif emotion == "adventure":
            prompt = f"Make this sound like an exciting and bold journey: {text}"
        elif emotion == "humor":
            prompt = f"Make this funny and engaging for social media: {text}"

    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
    output = t5_model.generate(input_ids, max_length=50)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)

# Caption generation
def generate_captions(image):
    inputs = blip_processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)

    # Random emotions
    old_emotion = random.choice(older_emotions)
    modern_emotion = random.choice(modern_emotions)

    # Styled captions
    older_caption = transform_caption(caption, "older", old_emotion)
    modern_caption = transform_caption(caption, "modern", modern_emotion)

    return older_caption, old_emotion, modern_caption, modern_emotion

# UI Setup
st.set_page_config(page_title="🖼️ Image Captioning", layout="centered")
st.title("🖼️ Image Captioning with Emotion and Style")
st.write("Upload an image to generate **older** and **modern** captions with different emotional tones.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button(" Generate Captions"):
        with st.spinner("Generating..."):
            older_caption, old_emotion, modern_caption, modern_emotion = generate_captions(image)

        st.markdown("### Older Style Caption")
        st.write(f"**Emotion:** {old_emotion.capitalize()}")
        st.success(older_caption)

        st.markdown("## Modern Style Caption")
        st.write(f"**Emotion:** {modern_emotion.capitalize()}")
        st.info(modern_caption)
