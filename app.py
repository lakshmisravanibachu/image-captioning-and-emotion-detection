import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_models()

# Simple rule-based emotion detection (improvement placeholder)
def detect_emotion(caption):
    caption = caption.lower()
    if any(word in caption for word in ["sunset", "nostalgic", "golden", "memories", "glow"]):
        return "Nostalgic"
    elif any(word in caption for word in ["happy", "fun", "smile", "friends", "joy"]):
        return "Happy"
    elif any(word in caption for word in ["storm", "dark", "lonely", "cold", "fear"]):
        return "Sad"
    elif any(word in caption for word in ["peace", "calm", "quiet", "relax", "serene"]):
        return "Peaceful"
    elif any(word in caption for word in ["wow", "amazing", "incredible", "shocking"]):
        return "Surprised"
    else:
        return "Neutral"

# Caption generator
def generate_caption(image, style="old"):
    raw_image = image.convert("RGB")

    # Style prompt
    if style == "old":
        prompt = "Describe this image in an old-fashioned, poetic tone."
    else:
        prompt = "Describe this image in a modern and casual tone."

    # Generate caption
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    emotion = detect_emotion(caption)

    return caption, emotion

# Streamlit UI
st.set_page_config(page_title="🖼️ Image Captioning", layout="centered")
st.title("🖼️ Image Captioning App")
st.write("Upload an image to generate both **older** and **modern** style captions with emotions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("✨ Generate Captions & Emotions"):
        st.subheader("Generating Captions...")

        # Generate styled captions
        old_caption, old_emotion = generate_caption(image, style="old")
        modern_caption, modern_emotion = generate_caption(image, style="modern")

        # Show results
        st.markdown("### 🕰️ Older Style")
        st.markdown(f"**Caption:** _{old_caption}_")
        st.markdown(f"**Emotion:** `{old_emotion}`")

        st.markdown("### 🧠 Modern Style")
        st.markdown(f"**Caption:** _{modern_caption}_")
        st.markdown(f"**Emotion:** `{modern_emotion}`")

