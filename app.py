import streamlit as st
from PIL import Image
import pytesseract
import speech_recognition as sr
from transformers import pipeline as transformers_pipeline
import tempfile
import os
import pytesseract
from PIL import Image

# Correct path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Example image processing
img = Image.open("ChatGPT Image Jul 7, 2025, 02_38_13 PM.png")  # Replace with your image file
text = pytesseract.image_to_string(img)

print("Extracted Text:", text)



# ✅ Set page config at the very top (important)
st.set_page_config(page_title="Multimodal AI Assistant", layout="centered")

# Load text-to-text generation model
@st.cache_resource
def load_model():
    return transformers_pipeline("text2text-generation", model="google/flan-t5-base")

nlp_pipeline = load_model()

# OCR: Extract text from image
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

# Audio: Transcribe audio file using Google's recognizer
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# UI
st.title(" Multimodal AI Assistant")
st.markdown("Upload an **image or audio file**, and type a question or `'summarize'`.")

uploaded_file = st.file_uploader("Upload Image or Audio", type=["png", "jpg", "jpeg", "wav"])
user_prompt = st.text_input("Ask a question or type 'summarize'")

if uploaded_file and user_prompt:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if file_ext in [".png", ".jpg", ".jpeg"]:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            extracted_text = extract_text_from_image(tmp_path)
        elif file_ext == ".wav":
            st.audio(uploaded_file)
            extracted_text = transcribe_audio(tmp_path)
        else:
            st.error("Unsupported file format.")
            extracted_text = ""

        if extracted_text.strip():
            st.markdown("### 📜 Extracted Text:")
            st.info(extracted_text.strip())

            # Prompt model
            if user_prompt.strip().lower() == "summarize":
                model_input = f"summarize: {extracted_text.strip()}"
            else:
                model_input = f"{user_prompt.strip()} Context: {extracted_text.strip()}"

            with st.spinner("🧠 Thinking..."):
                result = nlp_pipeline(model_input, max_length=200, do_sample=True)[0]['generated_text']
                st.markdown("### ✅ AI Response:")
                st.success(result)

        else:
            st.warning("❌ No text extracted.")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
