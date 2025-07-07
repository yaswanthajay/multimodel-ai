import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np
import speech_recognition as sr
from transformers import pipeline
import tempfile
import os

st.title("🎤🖼 Multimodal AI Assistant")

# OCR Function
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    text = pytesseract.image_to_string(img_array)
    return text

# Speech-to-Text Function
def transcribe_audio(file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

# Q&A pipeline
qa_pipeline = pipeline("question-answering")

# Image Input
st.subheader("📷 Image to Text (OCR)")
img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img:
    result = extract_text_from_image(img)
    st.text_area("Extracted Text", result, height=200)

# Audio Input
st.subheader("🎙️ Audio to Text")
audio_file = st.file_uploader("Upload an audio file", type=["wav"])
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_path = temp_audio.name
    try:
        transcription = transcribe_audio(temp_path)
        st.success("Transcription:")
        st.write(transcription)
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        os.remove(temp_path)

# Question Answering
st.subheader("💬 Ask a Question")
context = st.text_area("Context", "Multimodal AI combines audio, image, and text processing.")
question = st.text_input("Your question:")
if question:
    answer = qa_pipeline(question=question, context=context)
    st.write(f"Answer: {answer['answer']}")
