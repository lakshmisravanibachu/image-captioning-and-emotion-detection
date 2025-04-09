import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load models from local storage
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("./blip_processor")
    model = BlipForConditionalGeneration.from_pretrained("./blip_model")
    model.eval()
    return processor, model

processor, model = load_models()

def generate_caption(image, style="old"):
    raw_image = image.convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Dummy emotion based on style
    if style == "old":
        emotion = "Nostalgic"
    else:
        emotion = "Peaceful"

    return caption, emotion

# Streamlit UI
st.set_page_config(page_title="Image Captioning App", layout="centered")
st.title("🖼️ Image Captioning App")
st.write("Upload an image to generate both **older** and **modern** captions with emotions using the BLIP model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generating Captions...")
    old_caption, old_emotion = generate_caption(image, style="old")
    new_caption, new_emotion = generate_caption(image, style="modern")

    st.markdown("### 🕰️ Older Style")
    st.markdown(f"**Caption:** {old_caption}")
    st.markdown(f"**Emotion:** {old_emotion}")

    st.markdown("### 🧠 Modern Style")
    st.markdown(f"**Caption:** {new_caption}")
    st.markdown(f"**Emotion:** {new_emotion}")
