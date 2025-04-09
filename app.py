import streamlit as st
from PIL import Image
import numpy as np

# Function to generate captions and emotions
def generate_captions_and_emotions(image):
    # Preprocess the image
    img_tensor = read_image(image)
    img_features = caption_model.cnn_model(img_tensor)
    encoded_img = caption_model.encoder(img_features, training=False)

    # Generate Caption
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token

    final_caption = (
        decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    )

    # Generate Emotion (using the emotion detection model)
    img_array = np.array(image.resize((48, 48))).reshape(1, 48, 48, 1) / 255.0
    emotion_prediction = model.predict(img_array)
    emotion_index = np.argmax(emotion_prediction)
    detected_emotion = emotion_labels[emotion_index]

    return final_caption, detected_emotion

# Streamlit UI
st.title("Image Captioning and Emotion Detection")
st.write("Upload an image to get captions and detected emotions.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Generate captions and emotions
    caption, emotion = generate_captions_and_emotions(image)

    # Display results
    st.write("### Generated Caption:")
    st.write(caption)
    st.write("### Detected Emotion:")
    st.write(emotion)

# Run the Streamlit app
if __name__ == "__main__":
    st.run()
