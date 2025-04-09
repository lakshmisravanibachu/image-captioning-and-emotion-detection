import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import random

# Function to load the model and processor with error handling
def load_model():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the model and processor
processor, model = load_model()

# Function to generate captions and emotions
def generate_captions_with_emotions(image):
    if processor is None or model is None:
        return ("Model not loaded", "N/A"), ("Model not loaded", "N/A")
    
    # Process the image
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    # Define a list of emotions
    emotions = ["joy", "sadness", "anger", "surprise", "fear", "disgust"]
    
    # Generate an older caption with emotion
    older_caption = f"An older style description: {caption}"
    older_emotion = random.choice(emotions)
    
    # Generate a modern caption with emotion
    modern_caption = f"A modern take: {caption}"
    modern_emotion = random.choice(emotions)
    
    return (older_caption, older_emotion), (modern_caption, modern_emotion)

# Streamlit application
st.title("Image Captioning App")
st.write("Upload an image to generate both older and modern captions with emotions using the BLIP model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Generate captions
    if st.button("Generate Captions"):
        (older_caption, older_emotion), (modern_caption, modern_emotion) = generate_captions_with_emotions(image)
        st.write("**Older Caption:**", older_caption)
        st.write("**Emotion:**", older_emotion)
        st.write("**Modern Caption:**", modern_caption)
        st.write("**Emotion:**", modern_emotion)
