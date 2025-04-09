import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model and processor from Hugging Face
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_models()

def generate_caption(image, style="old"):
    raw_image = image.convert("RGB")
    
    # Use different prompts to influence style
    if style == "old":
        prompt = "Provide a vintage, poetic description for this image:"
    else:
        prompt = "Give a modern and casual description for this image:"
    
    # Process inputs with both image and text prompt
    inputs = processor(raw_image, text=prompt, return_tensors="pt")
    
    # Generate with sampling to reduce copy behavior
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    # Dummy emotion placeholders (replace with your own emotion detection if available)
    emotion = "Nostalgic" if style == "old" else "Peaceful"
    
    return caption, emotion

# Streamlit user interface
st.set_page_config(page_title="🖼️ Image Captioning with Emotion", layout="centered")
st.title("🖼️ Image Captioning with Emotion Detection")
st.write(
    "Upload an image and click **Generate Captions & Emotions** to get two different caption styles with emotions."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate captions only when the button is clicked
    if st.button("✨ Generate Captions & Emotions"):
        st.info("Generating Captions...")
        
        old_caption, old_emotion = generate_caption(image, style="old")
        modern_caption, modern_emotion = generate_caption(image, style="modern")

        st.markdown("### Older Style")
        st.markdown(f"**Caption:** {old_caption}")
        st.markdown(f"**Emotion:** {old_emotion}")

        st.markdown("### Modern Style")
        st.markdown(f"**Caption:** {modern_caption}")
        st.markdown(f"**Emotion:** {modern_emotion}")
