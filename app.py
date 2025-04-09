import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load from Hugging Face (works on Streamlit Cloud too)
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_models()

# Captioning function with style-specific prompts
def generate_caption(image, style="old"):
    raw_image = image.convert('RGB')

    # Prompt based on style
    if style == "old":
        prompt = "describe this image in an old-fashioned poetic way"
    else:
        prompt = "describe this image in a modern and casual style"

    inputs = processor(raw_image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50, num_beams=5)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Placeholder emotions
    if style == "old":
        emotion = "Nostalgic"
    else:
        emotion = "Peaceful"

    return caption, emotion

# Streamlit UI
st.set_page_config(page_title="🖼️ Image Captioning", layout="centered")
st.title("🖼️ Image Captioning with Emotion")
st.write("Upload an image to generate **older** and **modern** captions with emotion 🌟")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generating Captions...")

    old_caption, old_emotion = generate_caption(image, style="old")
    new_caption, new_emotion = generate_caption(image, style="modern")

    st.markdown("### Older Caption")
    st.markdown(f"**Emotion:** {old_emotion}")
    st.markdown(f"**Caption:** {old_caption}")

    st.markdown("### 🧠 Modern Caption")
    st.markdown(f"**Emotion:** {new_emotion}")
    st.markdown(f"**Caption:** {new_caption}")
