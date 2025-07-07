import streamlit as st
from PIL import Image
import pytesseract
import speech_recognition as sr
from transformers import pipeline
import tempfile
import os

# Load language model (summarizer or text generation)
nlp_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Extract text from image
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# Transcribe audio using Google's speech recognition
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# Streamlit UI
st.title("🧠 Multimodal AI Assistant")
st.markdown("Upload an **image or audio file**, and ask a question or let the model summarize.")

# File upload
uploaded_file = st.file_uploader("Upload Image or Audio", type=["png", "jpg", "jpeg", "wav"])

# Question input
user_prompt = st.text_input("Ask a question or type 'summarize'")

if uploaded_file is not None and user_prompt:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        if file_ext in [".png", ".jpg", ".jpeg"]:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            extracted_text = extract_text_from_image(temp_file_path)
        elif file_ext in [".wav"]:
            st.audio(uploaded_file)
            extracted_text = transcribe_audio(temp_file_path)
        else:
            st.error("Unsupported file format.")
            extracted_text = ""

        if extracted_text:
            st.markdown("**Extracted Text:**")
            st.write(extracted_text)

            # Prepare prompt
            if user_prompt.strip().lower() == "summarize":
                prompt = f"summarize: {extracted_text}"
            else:
                prompt = f"{user_prompt} Context: {extracted_text}"

            # Run model
            result = nlp_pipeline(prompt, max_length=200, do_sample=True)
            st.markdown("**AI Response:**")
            st.success(result[0]['generated_text'])
        else:
            st.warning("No text extracted.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
