# app.py
import streamlit as st
import pytesseract
import cv2
import tempfile
import os
import speech_recognition as sr
import pipeline

def extract_text_from_image(image_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_file.read())
        temp_path = temp_file.name

    image = cv2.imread(temp_path)
    text = pytesseract.image_to_string(image)
    os.remove(temp_path)
    return text

def extract_text_from_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Speech not recognized."
        except sr.RequestError:
            return "API unavailable."

def generate_response(prompt):
    response = nlp_pipeline(prompt, max_length=128, do_sample=True)
    return response[0]['generated_text']

st.title("🧠 Multimodal AI Assistant")

mode = st.radio("Choose your input type:", ("Text", "Image", "Audio"))

if mode == "Text":
    user_input = st.text_area("Enter your query:")
    if st.button("Generate Answer") and user_input:
        response = generate_response(user_input)
        st.success(response)

elif mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image and st.button("Extract & Generate"):
        extracted_text = extract_text_from_image(uploaded_image)
        st.write("**Extracted Text:**", extracted_text)
        if extracted_text:
            response = generate_response(extracted_text)
            st.success(response)

elif mode == "Audio":
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])
    if uploaded_audio and st.button("Transcribe & Generate"):
        transcribed_text = extract_text_from_audio(uploaded_audio)
        st.write("**Transcribed Text:**", transcribed_text)
        if transcribed_text:
            response = generate_response(transcribed_text)
            st.success(response)
