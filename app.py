
# app.py
import io
import os
from typing import List, Tuple
import threading

# ---- Optional: make TF safer on some CPUs before importing it
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ------------------------
# Config
# ------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "model.savedmodel")
DEFAULT_LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")
INPUT_SIZE: Tuple[int, int] = (224, 224)  # (height, width)
CROP_SIZE: int = 672

st.set_page_config(page_title="TM + TensorFlow Classifier", layout="centered")
st.title("Teachable Machine Model — Streamlit Web App")
st.caption("Runs a TensorFlow SavedModel on camera snapshots or a live browser camera stream.")

# ------------------------
# Helpers
# ------------------------
@st.cache_data(show_spinner=False)
def load_labels_from_path(labels_path: str) -> List[str]:
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

@st.cache_data(show_spinner=False)
def load_labels_from_uploaded(uploaded) -> List[str]:
    data = uploaded.read().decode("utf-8")
    labels = [line.strip() for line in data.splitlines() if line.strip()]
    return labels

@st.cache_resource(show_spinner=True)
def load_model(dir_path: str):
    # Loads TF SavedModel and returns the serving_default callable
    model = tf.saved_model.load(dir_path)
    infer = model.signatures["serving_default"]
    return infer

def preprocess_bgr_frame_for_model(frame_bgr: np.ndarray) -> np.ndarray:
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input
    img_rgb = cv2.resize(img_rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    # Normalize to [0, 1] float32
    img = img_rgb.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(img, axis=0)

def safe_center_crop_bgr(frame_bgr: np.ndarray, crop_size: int) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    tlx = max(0, cx - half)
    tly = max(0, cy - half)
    brx = min(w, tlx + crop_size)
    bry = min(h, tly + crop_size)
    # Adjust tl so that the crop is exactly crop_size if possible
    tlx = max(0, brx - crop_size)
    tly = max(0, bry - crop_size)
    return frame_bgr[tly:bry, tlx:brx], (tlx, tly), (brx, bry)

def predict(frame_bgr: np.ndarray, infer, labels: List[str]):
    x = preprocess_bgr_frame_for_model(frame_bgr)
    outputs = infer(tf.constant(x))
    # Get the first tensor in outputs dict robustly
    if isinstance(outputs, dict):
        for preferred in ("probabilities", "softmax", "predictions", "Identity", "output_0"):
            if preferred in outputs:
                probs = outputs[preferred].numpy()
                break
        else:
            probs = next(iter(outputs.values())).numpy()
    else:
        probs = outputs.numpy()

    probs = probs[0]
    top_idx = int(np.argmax(probs))
    label = labels[top_idx] if 0 <= top_idx < len(labels) else str(top_idx)
    score = float(probs[top_idx])
    return label, score, probs

# ------------------------
# Sidebar — Model & Labels
# ------------------------
with st.sidebar:
    st.header("Settings")
    model_dir = st.text_input("Model directory", value=MODEL_DIR, help="Folder containing saved_model.pb")
    labels_choice = st.radio("Labels source", ["Use labels.txt on disk", "Upload labels file"], index=0)
    uploaded_labels = None
    if labels_choice == "Upload labels file":
        uploaded_labels = st.file_uploader("Upload labels.txt", type=["txt"], accept_multiple_files=False)
    crop_size = st.slider("Crop size (pixels)", min_value=128, max_value=1024, value=CROP_SIZE, step=32)

# Load model
try:
    infer = load_model(model_dir)
except Exception as e:
    st.error(f"Failed to load model from '{model_dir}'. Make sure the folder exists and contains saved_model.pb. Error: {e}")
    st.stop()

# Load labels
try:
    if uploaded_labels is not None:
        labels = load_labels_from_uploaded(uploaded_labels)
    else:
        labels = load_labels_from_path(DEFAULT_LABELS_PATH)
except Exception as e:
    st.warning(f"Could not load labels. Falling back to indices. Error: {e}")
    labels = []

# A small lock for TF inference if multiple frames come concurrently
infer_lock = threading.Lock()

# ------------------------
# Browser Camera Snapshot
# ------------------------
st.subheader("Browser Camera Snapshot")
st.caption("Use your browser camera to capture an image and run inference.")
photo = st.camera_input("Capture a photo")
if photo is not None:
    image = Image.open(io.BytesIO(photo.getvalue())).convert("RGB")
    frame_rgb = np.array(image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    crop_bgr, (tlx, tly), (brx, bry) = safe_center_crop_bgr(frame_bgr, crop_size)
    with infer_lock:
        label, score, probs = predict(crop_bgr, infer, labels)

    vis = frame_bgr.copy()
    cv2.rectangle(vis, (tlx, tly), (brx, bry), (0, 255, 0), 3)
    cv2.putText(vis, f"{label} ({score:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    st.image(vis_rgb, caption=f"Prediction: {label} (confidence {score:.2f})", use_column_width=True)

    if len(labels) > 1 and probs is not None:
        import pandas as pd  # lazy import
        df = pd.DataFrame({"標籤 Label": labels, "機率 Probability": probs.astype(float)})
        df = df.sort_values(by="機率 Probability", ascending=False)
        st.dataframe(df, use_container_width=True)

# ------------------------
# Live Webcam (Phone/Browser) via WebRTC
# ------------------------
st.divider()
st.subheader("Live Webcam (Phone/Browser)")
st.caption(
    "Works in mobile browsers using WebRTC. For iOS, open in Safari and ensure HTTPS. "
    "Click 'Allow' when the browser asks for camera permissions."
)

# Pick which camera to use; 'environment' is rear camera on phones
cam_choice = st.radio("Camera", ["Rear (environment)", "Front (user)"], index=0)

video_constraints = {
    "video": {
        "facingMode": {"exact": "environment"} if cam_choice.startswith("Rear") else "user",
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 24},
    },
    "audio": False,
}

def live_webrtc_section(crop_size: int, infer, labels: List[str], infer_lock: threading.Lock, constraints: dict):
    # Import here so the app can boot before touching av/aiortc/ffmpeg
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # For Cloud/NAT reliability add TURN here.
            # {"urls": ["turn:YOUR_TURN_HOST:3478"], "username": "user", "credential": "pass"},
        ]
    })

    class TMVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_label = None
            self.last_score = None

        def recv(self, frame):
            # Lazy import av only when frames start arriving
            import av
            img_bgr = frame.to_ndarray(format="bgr24")
            crop_bgr, (tlx, tly), (brx, bry) = safe_center_crop_bgr(img_bgr, crop_size)
            with infer_lock:
                label, score, _ = predict(crop_bgr, infer, labels)
            self.last_label, self.last_score = label, score
            vis = img_bgr.copy()
            cv2.rectangle(vis, (tlx, tly), (brx, bry), (0, 255, 0), 3)
            cv2.putText(vis, f"{label} ({score:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(vis, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="tm-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints=constraints,
        video_processor_factory=TMVideoProcessor,
        async_processing=True,
    )

    if webrtc_ctx and webrtc_ctx.video_processor and webrtc_ctx.video_processor.last_label:
        st.info(f"Live prediction: **{webrtc_ctx.video_processor.last_label}** "
                f"(confidence **{webrtc_ctx.video_processor.last_score:.2f}**)")

# 👉 Call the live section so it renders
live_webrtc_section(crop_size, infer, labels, infer_lock, video_constraints)
