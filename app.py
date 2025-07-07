import streamlit as st
import pytesseract
import cv2
import numpy as np
import speech_recognition as sr
from transformers import pipeline
import tempfile
import os

st.title("🤖 Multimodal AI Assistant")

task = st.selectbox("Select Task", ["Text from Image (OCR)", "Speech to Text", "Ask AI Question"])

if task == "Text from Image (OCR)":
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        text = pytesseract.image_to_string(img)
        st.image(img, caption="Uploaded Image", channels="BGR")
        st.subheader("📝 Extracted Text")
        st.write(text)

elif task == "Speech to Text":
    st.write("🎤 Record your voice (WAV only)")
    audio_file = st.file_uploader("Upload Audio File", type=["wav"])
    if audio_file:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                st.subheader("📝 Transcribed Text")
                st.write(text)
            except sr.UnknownValueError:
                st.error("Speech Recognition could not understand audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")

elif task == "Ask AI Question":
    question = st.text_input("Ask me anything:")
    if question:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        context = "This is a demo assistant using OCR, speech recognition, and NLP."
        result = qa_pipeline(question=question, context=context)
        st.subheader("🤖 AI Answer")
        st.write(result['answer'])
