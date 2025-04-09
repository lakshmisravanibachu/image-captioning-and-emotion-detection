import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import torch

# Load models only once
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    return blip_processor, blip_model, t5_tokenizer, t5_model

blip_processor, blip_model, t5_tokenizer, t5_model = load_models()

# Assign tone based on content
def detect_tone(caption, style="older"):
    caption = caption.lower()
    if style == "older":
        if any(word in caption for word in ["storm", "ship", "dark", "night", "soldier"]):
            return "danger"
        elif any(word in caption for word in ["old", "village", "memory", "classic", "black and white"]):
            return "nostalgia"
        elif any(word in caption for word in ["mystery", "fog", "shadow", "hidden"]):
            return "mystery"
        else:
            return "sadness"
    else:
        if any(word in caption for word in ["party", "dance", "smile", "happy"]):
            return "joy"
        elif any(word in caption for word in ["run", "ride", "jump", "race"]):
            return "excitement"
        elif any(word in caption for word in ["trip", "mountain", "beach", "explore"]):
            return "adventure"
        else:
            return "humor"

# Caption rewriter
def rewrite_caption(caption, style, emotion):
    prompt_map = {
        "older": {
            "nostalgia": "Rewrite this in an old-fashioned, sentimental style",
            "sadness": "Rewrite this as a melancholic and deep story",
            "mystery": "Rewrite this in a suspenseful and enigmatic tone",
            "danger": "Rewrite this in a thrilling and perilous manner",
        },
        "modern": {
            "joy": "Make this fun, cheerful, and lighthearted",
            "excitement": "Make this energetic and thrilling",
            "adventure": "Make this sound like an exciting and bold journey",
            "humor": "Make this funny and engaging for social media",
        },
    }
    prompt = f"{prompt_map[style][emotion]}: {caption}"
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
    output = t5_model.generate(input_ids, max_length=50)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)

# Get base caption
def get_base_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# Streamlit app
st.set_page_config(page_title="Dual Emotion Caption Generator")
st.title("🖼️ Image-Based Emotion Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating captions..."):
        base_caption = get_base_caption(image)

        # Determine emotions based on caption
        older_emotion = detect_tone(base_caption, style="older")
        modern_emotion = detect_tone(base_caption, style="modern")

        # Rewrite captions
        older_caption = rewrite_caption(base_caption, "older", older_emotion)
        modern_caption = rewrite_caption(base_caption, "modern", modern_emotion)

    st.markdown("### 🧠 Base Caption")
    st.info(base_caption)

    st.markdown(f"### 🕰️ Older Style Caption ({older_emotion.capitalize()})")
    st.success(older_caption)

    st.markdown(f"### ⚡ Modern Style Caption ({modern_emotion.capitalize()})")
    st.warning(modern_caption)
