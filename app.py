
import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
import io
import numpy as np
from PIL import Image
import base64

# Optional/lightweight deps
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero glow">
      <div class="pill">Global AI Buildathon</div>
      <div class="pill">Accessibility</div>
      <h1 class="fade-in">Able<span class="accent">AI</span> – Multi‑Modal Accessibility Assistant</h1>
      <p class="small">Convert <b>voice ↔ text</b>, <b>text → speech</b>, and generate <b>alt‑text for images</b>. Built for inclusive education & communication.</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("All features are designed to run with lightweight dependencies.\n\nIf a module is missing in your environment, the UI will degrade gracefully.")
    st.divider()
    hf_token = st.text_input("Optional: Hugging Face Inference API Token (for better image captions)", type="password", help="If provided, we will try calling a hosted image-caption model. Otherwise, we use an offline alt‑text heuristic.")
    st.caption("Tip: Leave blank if you don't have a token.")

tab1, tab2, tab3, tab4 = st.tabs(["🎤 Speech → Text", "🗣️ Text → Speech", "🖼️ Image → Alt‑Text", "🤟 Sign (Demo)"])

# --------------- HELPERS ---------------

def wav_bytes_from_micrecorder(rec):
    if not rec:
        return None
    data = rec.get("bytes")
    if data is None:
        b64 = rec.get("audio")
        if b64:
            return base64.b64decode(b64.split(",")[-1])
        return None
    return data

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

def run_text_to_speech(text):
    if gTTS is None:
        return None, "gTTS is not installed."
    if not text.strip():
        return None, "Please type some text."
    try:
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
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

# --------------- TAB 1: SPEECH → TEXT ---------------
with tab1:
    st.subheader("🎤 Speech to Text")
    colA, colB = st.columns([1,1])
    with colA:
        st.write("Record a short clip and transcribe it. Works best with clear speech.")
        rec = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop",
            just_once=False,
            use_container_width=True,
            format="wav",
        )
        if rec:
            st.success("Audio recorded. Click **Transcribe**.")
    with colB:
        st.audio(wav_bytes_from_micrecorder(rec) if rec else None, format="audio/wav")

    st.caption("Note: Uses Google's free web STT via the `SpeechRecognition` library. No key required, limited quota.")

    if st.button("🪄 Transcribe", use_container_width=True, type="primary"):
        if rec:
            wav_bytes = wav_bytes_from_micrecorder(rec)
            if wav_bytes:
                with st.spinner("Transcribing..."):
                    text, err = run_speech_to_text(wav_bytes)
                if err:
                    st.error(err)
                else:
                    st.text_area("Transcript", value=text, height=180)
            else:
                st.warning("No audio found from recorder.")
        else:
            st.warning("Please record audio first.")

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
        st.download_button("⬇️ Download MP3", data=(audio_bytes if audio_bytes else None), file_name="ableai_tts.mp3", disabled=not(bool(audio_bytes)))

# --------------- TAB 3: IMAGE → ALT-TEXT ---------------
with tab3:
    st.subheader("🖼️ Image to Alt‑Text / Caption")
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
            if st.button("⚡ Quick Alt‑Text (Offline)"):
                cap = simple_alt_text(img)
                st.info("Heuristic Alt‑Text:")
                st.write(cap)
        with colR:
            st.caption("Tip: Provide a Hugging Face token in the sidebar to use BLIP for high‑quality captions.")

# --------------- TAB 4: SIGN DEMO ---------------
with tab4:
    st.subheader("🤟 Sign Recognition (Lightweight Demo)")
    st.write("This demo runs **in the browser** using MediaPipe Hands (JavaScript) in an embedded view. It highlights hand landmarks and shows basic gesture cues (open palm / fist / thumbs‑up).")
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
          <span class="badge">Live demo • On‑device</span>
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
            if (extended === 1 && pts[4].x < pts[3].x) return 'Thumbs‑up 👍 (Yes)';
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
    st.caption("This browser demo avoids heavy Python dependencies and is deploy‑friendly. For richer ISL/ASL models (sequence classification), integrate a hosted model endpoint or a custom Mediapipe+LSTM pipeline later.")

st.markdown("<div class='footer'>Built with ❤️ for inclusive access. • AbleAI</div>", unsafe_allow_html=True)
