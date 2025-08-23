import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
import os
import tempfile
from transformers import pipeline
import streamlit_lottie as st_lottie
import requests
import io

# Lottie animation helper
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(page_title="AbleAI - Accessibility Assistant", layout="wide")
lottie_ai = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_mziehk.json")

st.title("🤖 AbleAI: AI-Powered Accessibility Assistant")
st_lottie.st_lottie(lottie_ai, speed=1, loop=True, quality="high", height=300)

st.sidebar.title("Choose Conversion")
feature = st.sidebar.selectbox(
    "Task",
    ['Speech-to-Text', 'Text-to-Speech', 'Sign Language to Text', 'Image Captioning']
)

if feature == 'Speech-to-Text':
    st.header("🎤 Speech-to-Text")
    audio_file = st.file_uploader("Upload an audio file (WAV/MP3)", type=['wav', 'mp3'])
    if st.button("Transcribe Audio"):
        if audio_file:
            recognizer = sr.Recognizer()
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp.flush()
                with sr.AudioFile(tmp.name) as source:
                    audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio)
                    st.success("Transcription:")
                    st.write(text)
                except Exception as e:
                    st.error(f"Recognition error: {e}")

elif feature == 'Text-to-Speech':
    st.header("🗣️ Text-to-Speech")
    text_input = st.text_area("Enter text to convert")
    lang = st.selectbox("Language", ['en', 'hi', 'fr', 'es']) # Add more lang codes if needed
    if st.button("Generate Voice"):
        if text_input:
            tts = gTTS(text_input, lang=lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                st.audio(fp.name, format='audio/mp3')
                st.download_button(
                    label="Download audio",
                    data=open(fp.name,'rb').read(),
                    file_name="output.mp3"
                )

elif feature == "Sign Language to Text":
    st.header("🤟 Sign Language to Text")
    st.write("Upload a hand sign image (eg. A-Z, ISL/ASL hand sign). [Demo only]")
    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, caption='Uploaded Sign Language Image', use_column_width=True)
        # --- Replace with your sign language model logic
        st.info("Demo prediction: 'Hello'")
        st.success("Predicted Text: Hello")
    st.caption("For real-world, train with ISL/ASL models & datasets.")

elif feature == 'Image Captioning':
    st.header("🖼️ Image Captioning (AI-powered)")
    # Using transformers pipeline for demo
    img_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Input Image", use_column_width=True)
        st.write("Generating caption...")
        captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        result = captioner(img, max_new_tokens=25)
        st.success(f"Caption: {result[0]['generated_text']}")

st.markdown("---")
st.info("Built for Global AI Buildathon. Adapted with inspiration from Ishaara (ISL translation), award-winning accessibility tools.")
st.caption("Made with ❤️. Streamlit | Python | Transformers | OpenAI | Lottie | Accessibility First.")
