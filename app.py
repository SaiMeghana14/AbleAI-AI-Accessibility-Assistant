import streamlit as st
import io
import numpy as np
from PIL import Image
import base64
import json
import math
import time
import datetime as dt

# Optional/lightweight deps
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

# Optional OCR (local); streamlit cloud may not have tesseract binary
try:
    import pytesseract
except Exception:
    pytesseract = None

# ---------------- SESSION / USAGE ----------------
def init_state():
    ss = st.session_state
    ss.setdefault("usage", {
        "stt": 0,
        "tts": 0,
        "ocr": 0,
        "summaries": 0,
        "listen_seconds": 0.0
    })
    ss.setdefault("badges", set())
    ss.setdefault("log", [])  # list of dict events
init_state()

def log_event(kind, meta=None):
    meta = meta or {}
    st.session_state["log"].append({
        "ts": time.time(),
        "kind": kind,
        "meta": meta
    })

def add_count(key, inc=1):
    st.session_state["usage"][key] = st.session_state["usage"].get(key, 0) + inc

def unlock_badges():
    u = st.session_state["usage"]
    before = set(st.session_state["badges"])
    if u["tts"] >= 5:
        st.session_state["badges"].add("🥉 First 5 Reads")
    if u["tts"] >= 25:
        st.session_state["badges"].add("🥈 25 Reads")
    if u["tts"] >= 100:
        st.session_state["badges"].add("🥇 100 Texts Read")
    if u["stt"] >= 10:
        st.session_state["badges"].add("🎤 Voice Pro")
    if u["ocr"] >= 10:
        st.session_state["badges"].add("📖 Reader Champion")
    if u["summaries"] >= 10:
        st.session_state["badges"].add("📝 Summarizer Star")
    if u["listen_seconds"] >= 3600:
        st.session_state["badges"].add("⏱️ 1 Hour Listened")
    if st.session_state["badges"] - before:
        st.balloons()

# --------------- THEME / PAGE CONFIG ---------------
st.set_page_config(
    page_title="AbleAI – Accessibility Assistant",
    page_icon="🧩",
    layout="wide",
    menu_items={"about": "AbleAI • AI-Powered Accessibility Assistant"}
)

# Subtle CSS & animations (no heavy libs)
st.markdown(
    """
    <style>
      @keyframes floaty { 0%{transform:translateY(0px)} 50%{transform:translateY(-6px)} 100%{transform:translateY(0px)} }
      .hero {padding: 14px 18px; border-radius: 18px; background: linear-gradient(135deg, #f1f5f9 0%, #f8fafc 100%);}
      .glow {box-shadow: 0 10px 30px rgba(0,0,0,0.08); border-radius: 18px;}
      .pill {display:inline-block; padding:6px 12px; border-radius:999px; background:#e2e8f0; font-size:12px; margin-right:6px;}
      .fade-in {animation: floaty 3s ease-in-out infinite;}
      .accent {background: linear-gradient(90deg,#7c3aed,#06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight:800;}
      .small {font-size:13px; opacity:.9}
      .footer {opacity:.6; font-size:12px; margin-top:6px}
      .badge {display:inline-block; padding:6px 10px; border-radius:999px; background:#eef2ff; margin-right:6px; margin-bottom:6px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero glow">
      <div class="pill">Global AI Buildathon</div>
      <div class="pill">Accessibility</div>
      <h1 class="fade-in">Able<span class="accent">AI</span> – Multi-Modal Accessibility Assistant</h1>
      <p class="small">Convert <b>voice ↔ text</b>, <b>text → speech</b>, generate <b>alt-text</b>, read text from images (OCR), summarize, and visualize your impact on accessibility.</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("All features are designed to run with lightweight dependencies. If a module is missing, the UI degrades gracefully.")
    st.divider()
    hf_token = st.text_input(
        "Optional: Hugging Face Inference API Token (captions + summaries)",
        type="password",
        help="If provided, we use hosted models for better image captions and summarization."
    )
    ocr_space_key = st.text_input(
        "Optional: OCR.space API Key",
        type="password",
        help="Fallback OCR if local Tesseract is unavailable."
    )
    st.caption("Tip: Leave blank if you don't have tokens/keys.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎤 Speech → Text", "🗣️ Text → Speech", "🖼️ Image → Alt-Text",
    "📷 OCR Reader", "📝 Summarizer", "🤟 Sign (Demo)", "📊 Dashboard"
])

# --------------- HELPERS ---------------
def run_speech_to_text(wav_bytes):
    if sr is None:
        return None, "SpeechRecognition is not installed."
    try:
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text, None
    except sr.UnknownValueError:
        return "", "Sorry, I couldn’t understand the audio."
    except Exception as e:
        return None, f"STT failed: {e}"

def run_text_to_speech(text, slow=False):
    if gTTS is None:
        return None, "gTTS is not installed."
    if not text.strip():
        return None, "Please type some text."
    try:
        tts = gTTS(text=text, lang="en", slow=slow)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        # assume 150 wpm ≈ 2.5 wps; rough duration estimate:
        est_seconds = max(2.5, len(text.split()) / 2.5) * (1.25 if slow else 1.0)
        st.session_state["usage"]["listen_seconds"] += float(est_seconds)
        add_count("tts", 1)
        log_event("tts", {"chars": len(text), "slow": slow})
        unlock_badges()
        return buf.read(), None
    except Exception as e:
        return None, f"TTS failed: {e}"

def simple_alt_text(img: Image.Image):
    im = img.convert("RGB")
    w, h = im.size
    arr = np.array(im)
    brightness = arr.mean()
    hsv = np.array(im.convert("HSV"))
    h_channel = hsv[...,0]
    hist, _ = np.histogram(h_channel.flatten(), bins=12, range=(0,255))
    dom_idx = int(hist.argmax())
    hue_names = ["red","orange","yellow","lime","green","teal","cyan","azure","blue","violet","magenta","rose"]
    dominant = hue_names[dom_idx % len(hue_names)]
    orient = "landscape" if w > h else ("portrait" if h > w else "square")
    bright_label = "bright" if brightness > 170 else ("dim" if brightness < 90 else "moderately lit")
    caption = f"A {orient} image ({w}×{h}px) with {bright_label} lighting and dominant {dominant} tones."
    return caption

def call_hf_caption(img_bytes, token):
    import requests
    url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(url, headers=headers, data=img_bytes, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"], None
        if isinstance(data, dict) and data.get("error") and "loading" in data["error"].lower():
            return None, "Model is warming up on HF. Try again shortly."
        return None, f"Unexpected HF response: {data}"
    except Exception as e:
        return None, f"HF caption error: {e}"

# --- OCR helpers ---
def ocr_local(image: Image.Image):
    if pytesseract is None:
        return None, "Local Tesseract not available."
    try:
        text = pytesseract.image_to_string(image)
        return text.strip(), None
    except Exception as e:
        return None, f"OCR error: {e}"

def ocr_remote_ocrspace(img_bytes, apikey=None):
    import requests
    try:
        url = "https://api.ocr.space/parse/image"
        payload = {
            "OCREngine": 2,
            "scale": True,
            "isOverlayRequired": False,
            "language": "eng"
        }
        headers = {}
        if apikey:
            headers["apikey"] = apikey
        files = {"file": ("image.png", img_bytes, "application/octet-stream")}
        resp = requests.post(url, data=payload, files=files, headers=headers, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        if data.get("IsErroredOnProcessing"):
            return None, data.get("ErrorMessage", "OCR.space error")
        parsed = data.get("ParsedResults", [])
        if parsed:
            return parsed[0].get("ParsedText", "").strip(), None
        return None, "No text found."
    except Exception as e:
        return None, f"OCR.space error: {e}"

# --- Summarizer helpers ---
def summarize_hf(text, token):
    import requests
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": text, "parameters": {"min_length": 60, "max_length": 180}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "summary_text" in data[0]:
            return data[0]["summary_text"].strip(), None
        if isinstance(data, dict) and data.get("error"):
            return None, data.get("error")
        return None, f"Unexpected HF response: {data}"
    except Exception as e:
        return None, f"HF summarization error: {e}"

def summarize_local(text, max_sentences=5):
    # ultra-simple frequency-based extractive summarizer
    import re
    from collections import Counter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return text.strip(), None
    words = re.findall(r'\w+', text.lower())
    stop = set("""
        a an the and or of in to for with on at from into during including until against
        among throughout despite towards upon about above below over under again further then once here there
        all any both each few more most other some such no nor not only own same so than too very
        can will just don should now is are was were be been being have has had do does did
    """.split())
    words = [w for w in words if w not in stop and len(w) > 2]
    scores = Counter(words)
    sent_scores = []
    for s in sentences:
        tokens = re.findall(r'\w+', s.lower())
        score = sum(scores.get(t, 0) for t in tokens) / (len(tokens) + 1e-6)
        sent_scores.append((score, s))
    top = sorted(sent_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_sents = [s for _, s in sorted(top, key=lambda x: sentences.index(x[1]))]
    return " ".join(top_sents).strip(), None

# --- Emotion (very lightweight heuristics on WAV/AIFF/FLAC) ---
def detect_emotion_from_wav(wav_bytes):
    """
    Very small, on-device heuristic:
    - Compute RMS energy & zero-crossing rate on raw mono signal
    - Classify: 'stressed' if high energy + high ZCR, 'calm' if low energy, else 'neutral'
    """
    import wave
    import struct
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            fr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
        # unpack to ints
        if sampwidth == 2:
            fmt = "<" + "h" * (len(raw)//2)
            data = np.array(struct.unpack(fmt, raw), dtype=np.float32)
        else:
            # fallback: treat bytes as unsigned then center
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        # normalize
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        # RMS & ZCR
        rms = np.sqrt(np.mean(data**2))
        zc = np.mean(np.abs(np.diff(np.sign(data))))  # approx zcr
        # thresholds from quick heuristics
        if rms > 0.25 and zc > 0.18:
            return "stressed", {"rms": float(rms), "zcr": float(zc)}
        if rms < 0.08:
            return "calm", {"rms": float(rms), "zcr": float(zc)}
        return "neutral", {"rms": float(rms), "zcr": float(zc)}
    except Exception as e:
        return "unknown", {"error": str(e)}

# --------------- TAB 1: SPEECH → TEXT ---------------
with tab1:
    st.subheader("🎤 Speech-to-Text")
    st.caption("Upload **WAV/AIFF/FLAC** for best results. MP3 isn’t supported by the STT backend here.")
    uploaded_audio = st.file_uploader("Upload audio (WAV/AIFF/FLAC)", type=["wav", "aiff", "flac"])
    
    transcript_container = st.empty()
    emotion_container = st.empty()

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        wav_bytes = uploaded_audio.read()

        # Emotion (lightweight, heuristic)
        emo, emo_meta = detect_emotion_from_wav(wav_bytes)
        if emo == "unknown":
            emotion_container.warning(f"Emotion detection unavailable: {emo_meta.get('error','')}")
        else:
            emotion_container.info(f"Detected emotion: **{emo}**  ·  (RMS={emo_meta.get('rms',0):.3f}, ZCR={emo_meta.get('zcr',0):.3f})")

        if st.button("🪄 Transcribe", use_container_width=True, type="primary"):
            with st.spinner("Transcribing..."):
                text, err = run_speech_to_text(wav_bytes)
            if err:
                st.error(err)
            else:
                add_count("stt", 1)
                log_event("stt", {"chars": len(text)})
                unlock_badges()
                transcript_container.text_area("Transcript", value=text, height=180)

                # Optional read-back with emotion-aware speed
                if gTTS is not None and text.strip():
                    slow_flag = (emo == "stressed")
                    if st.button("🔁 Read Transcript Aloud (emotion-aware)"):
                        audio_bytes, err_tts = run_text_to_speech(text, slow=slow_flag)
                        if err_tts:
                            st.error(err_tts)
                        else:
                            st.audio(audio_bytes, format="audio/mp3")
                            st.download_button("⬇️ Download MP3", data=audio_bytes, file_name="transcript_tts.mp3", mime="audio/mp3")

# --------------- TAB 2: TEXT → SPEECH ---------------
with tab2:
    st.subheader("🗣️ Text to Speech")
    tts_text = st.text_area("Enter text", placeholder="Type or paste text to convert into natural speech...")
    col1, col2 = st.columns([1,1])
    audio_bytes = None
    with col1:
        if st.button("🔊 Synthesize", type="primary", use_container_width=True):
            with st.spinner("Generating speech..."):
                audio_bytes, err = run_text_to_speech(tts_text)
            if err:
                st.error(err)
            else:
                st.audio(audio_bytes, format="audio/mp3")

    with col2:
        if audio_bytes:
            st.download_button(
                "⬇️ Download MP3",
                data=audio_bytes,
                file_name="ableai_tts.mp3",
                mime="audio/mp3"
            )
        else:
            st.info("⚠️ Generate some audio to enable download.")

# --------------- TAB 3: IMAGE → ALT-TEXT ---------------
with tab3:
    st.subheader("🖼️ Image to Alt-Text / Caption")
    up = st.file_uploader("Upload an image (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=False)
    if up:
        img = Image.open(up)
        st.image(img, caption="Preview", use_column_width=True)
        colL, colR = st.columns([1,1])
        with colL:
            if hf_token:
                if st.button("🧠 Generate Caption (Hugging Face)", type="primary"):
                    with st.spinner("Calling Hugging Face caption model..."):
                        cap, err = call_hf_caption(up.getvalue(), hf_token)
                    if err:
                        st.error(err)
                    elif cap:
                        st.success("Caption (HF):")
                        st.write(cap)
            if st.button("⚡ Quick Alt-Text (Offline)"):
                cap = simple_alt_text(img)
                st.info("Heuristic Alt-Text:")
                st.write(cap)
        with colR:
            st.caption("Tip: Provide a Hugging Face token in the sidebar to use BLIP for high-quality captions.")

# --------------- TAB 4: OCR READER ---------------
with tab4:
    st.subheader("📷 OCR Reader (Image → Text → Speech)")
    img_up = st.file_uploader("Upload an image with text (PNG/JPG)", type=["png","jpg","jpeg"])
    if img_up:
        img = Image.open(img_up)
        st.image(img, caption="OCR Preview", use_column_width=True)

        ocr_text = ""
        run_local = st.checkbox("Try local OCR (Tesseract) first", value=True, help="Requires pytesseract + tesseract binary")
        if st.button("🔎 Extract Text"):
            with st.spinner("Running OCR..."):
                if run_local and pytesseract is not None:
                    ocr_text, err = ocr_local(img)
                    if err:
                        st.warning(f"Local OCR fallback needed: {err}")
                        ocr_text = ""
                if not ocr_text:
                    # fallback remote
                    ocr_text, err = ocr_remote_ocrspace(img_up.getvalue(), apikey=ocr_space_key or None)
                    if err:
                        st.error(err)
                if ocr_text:
                    add_count("ocr", 1)
                    log_event("ocr", {"chars": len(ocr_text)})
                    unlock_badges()
                    st.success("OCR Result")
                    st.text_area("Extracted Text", ocr_text, height=220)
        if gTTS is not None and st.button("🔊 Read Extracted Text"):
            if ocr_text:
                audio_bytes, err = run_text_to_speech(ocr_text)
                if err:
                    st.error(err)
                else:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button("⬇️ Download MP3", data=audio_bytes, file_name="ocr_readout.mp3", mime="audio/mp3")
            else:
                st.info("Run OCR first to get text.")

# --------------- TAB 5: SUMMARIZER ---------------
with tab5:
    st.subheader("📝 Smart Summarizer")
    st.caption("Paste long text to summarize. Uses HF (if token) or a lightweight local summarizer.")
    long_text = st.text_area("Input text", height=220, placeholder="Paste or type long text here...")
    if st.button("✨ Summarize", type="primary"):
        if not long_text.strip():
            st.warning("Please paste some text.")
        else:
            with st.spinner("Summarizing..."):
                if hf_token:
                    summary, err = summarize_hf(long_text, hf_token)
                else:
                    summary, err = summarize_local(long_text)
            if err:
                st.error(err)
            else:
                add_count("summaries", 1)
                log_event("summary", {"chars": len(long_text)})
                unlock_badges()
                st.success("Summary")
                st.write(summary)
                if gTTS is not None and st.toggle("🔊 Read summary aloud"):
                    audio_bytes, err = run_text_to_speech(summary)
                    if not err and audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                        st.download_button("⬇️ Download MP3", data=audio_bytes, file_name="summary_tts.mp3", mime="audio/mp3")

# --------------- TAB 6: SIGN DEMO ---------------
with tab6:
    st.subheader("🤟 Sign Recognition (Lightweight Demo)")
    st.write("This demo runs **in the browser** using MediaPipe Hands (JavaScript). It highlights hand landmarks and shows basic gesture cues (open palm / fist / thumbs-up).")
    import streamlit.components.v1 as components
    components.html(
        """
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
         body{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI; margin:0}
         #wrap{display:flex; flex-direction:column; gap:8px; padding:8px}
         video, canvas{width:100%; max-width:720px; border-radius:16px; box-shadow: 0 10px 30px rgba(0,0,0,.08)}
         .badge{display:inline-block; padding:6px 10px; border-radius:999px; background:#eef2ff}
        </style>
        </head>
        <body>
        <div id="wrap">
          <span class="badge">Live demo • On-device</span>
          <video id="video" autoplay playsinline></video>
          <canvas id="out"></canvas>
          <div id="label" class="badge">Initializing…</div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js"></script>
        <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('out');
        const ctx = canvas.getContext('2d');
        const label = document.getElementById('label');

        async function main(){
          const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
          const handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task' },
            numHands: 1,
            runningMode: 'VIDEO'
          });
          const stream = await navigator.mediaDevices.getUserMedia({video:true});
          video.srcObject = stream;
          await new Promise(r => video.onloadedmetadata = r);
          video.play();
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          function gestureFromLandmarks(lms){
            if(!lms || !lms.length) return 'No hand';
            const pts = lms[0];
            const tipIdx = [4,8,12,16,20];
            const wristY = pts[0].y;
            let extended = 0;
            for (const i of tipIdx){
              if (pts[i].y < wristY - 0.05) extended += 1;
            }
            if (extended >= 4) return 'Open palm 👋 (Hello)';
            if (extended === 0) return 'Fist ✊ (Stop)';
            if (extended === 1 && pts[4].x < pts[3].x) return 'Thumbs-up 👍 (Yes)';
            return 'Neutral hand';
          }

          async function loop(){
            const now = performance.now();
            const res = await handLandmarker.detectForVideo(video, now);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            if (res && res.handednesses && res.handednesses.length>0){
              const lms = res.landmarks;
              for (const points of lms){
                ctx.fillStyle = '#10b981';
                for (const p of points){
                  ctx.beginPath();
                  ctx.arc(p.x*canvas.width, p.y*canvas.height, 4, 0, Math.PI*2);
                  ctx.fill();
                }
              }
              label.textContent = gestureFromLandmarks(res.landmarks);
            }else{
              label.textContent = 'Show your hand to the camera';
            }
            requestAnimationFrame(loop);
          }
          loop();
        }
        main().catch(e => { label.textContent = 'Error: '+e; });
        </script>
        </body>
        </html>
        """,
        height=720
    )
    st.caption("This browser demo avoids heavy Python dependencies and is deploy-friendly. For richer ISL/ASL models (sequence classification), integrate a hosted model endpoint or a custom Mediapipe+LSTM pipeline later.")

# --------------- TAB 7: DASHBOARD ---------------
with tab7:
    st.subheader("📊 Accessibility Dashboard")
    u = st.session_state["usage"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔊 Text → Speech", u["tts"])
    c2.metric("🎤 Speech → Text", u["stt"])
    c3.metric("📷 OCR Reads", u["ocr"])
    c4.metric("📝 Summaries", u["summaries"])
    st.caption(f"⏱️ Time listened (est.): **{int(u['listen_seconds']//60)} min {int(u['listen_seconds']%60)} s**")

    # Charts (matplotlib-free Streamlit built-ins to keep deps minimal)
    import pandas as pd
    usage_df = pd.DataFrame({
        "Feature": ["TTS","STT","OCR","Summaries"],
        "Count": [u["tts"], u["stt"], u["ocr"], u["summaries"]]
    })
    st.bar_chart(usage_df.set_index("Feature"))

    # Timeline (last 50 events)
    if st.session_state["log"]:
        tl = pd.DataFrame(st.session_state["log"][-50:])
        tl["ts"] = tl["ts"].apply(lambda x: dt.datetime.fromtimestamp(x).strftime("%H:%M:%S"))
        st.line_chart(tl.groupby("ts").size())

    st.markdown("### 🏆 Badges")
    if st.session_state["badges"]:
        for b in sorted(st.session_state["badges"]):
            st.markdown(f"<span class='badge'>{b}</span>", unsafe_allow_html=True)
    else:
        st.info("Use the features to unlock badges!")

st.markdown("<div class='footer'>Built with ❤️ for inclusive access. • AbleAI</div>", unsafe_allow_html=True)
