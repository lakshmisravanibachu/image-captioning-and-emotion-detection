import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the BLIP model and processor from Hugging Face
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_models()

def generate_caption(image, style="old"):
    raw_image = image.convert("RGB")
    
    # Extended prompts including example styles
    if style == "old":
        prompt = (
            "Compose a caption for the image that is in a classical poetic style. "
            "For example: “A golden orb sinks beyond the silent hills, casting hues of amber upon the tranquil land.” "
            "Now provide a similar style description for the image:"
        )
    else:
        prompt = (
            "Compose a caption for the image that is in a modern, casual style. "
            "For example: “Beautiful sunset with warm colors lighting up the sky!” "
            "Now provide a similar style description for the image:"
        )
    
    # Combine the image with the text prompt
    inputs = processor(raw_image, text=prompt, return_tensors="pt")
    
    # Generate caption using sampling for diversity
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=60,
            num_beams=5,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
    
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    
    # Dummy emotion labels for now
    emotion = "Nostalgic" if style == "old" else "Peaceful"
    
    return caption, emotion

# Streamlit UI
st.set_page_config(page_title="🖼️ Image Captioning with Emotion", layout="centered")
st.title("🖼️ Image Captioning with Emotion Detection")
st.write("Upload an image below and click **Generate Captions & Emotions** to see two style-specific captions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("✨ Generate Captions & Emotions"):
        st.info("Generating captions, please wait...")
        
        # Generate old and modern style captions
        old_caption, old_emotion = generate_caption(image, style="old")
        modern_caption, modern_emotion = generate_caption(image, style="modern")
        
        st.markdown("### 🕰️ Older Style")
        st.markdown(f"**Caption:** {old_caption}")
        st.markdown(f"**Emotion:** {old_emotion}")
        
        st.markdown("### 🧠 Modern Style")
        st.markdown(f"**Caption:** {modern_caption}")
        st.markdown(f"**Emotion:** {modern_emotion}")
