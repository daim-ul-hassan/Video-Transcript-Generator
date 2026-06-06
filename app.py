import base64
import os
import tempfile
from io import BytesIO

import cv2
import streamlit as st
from groq import Groq
from PIL import Image


st.set_page_config(page_title="Video Scene Describer", page_icon="video", layout="centered")

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def get_groq_client() -> Groq:
    api_key = st.session_state.get("groq_api_key", "").strip()
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        secret_key = None
    api_key = api_key or secret_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to your environment or Streamlit secrets.")
    return Groq(api_key=api_key)


def resize_frame(frame_bgr, max_width: int = 1024) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    if image.width > max_width:
        new_height = int(image.height * (max_width / image.width))
        image = image.resize((max_width, new_height))
    return image


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def extract_frames_every_n_seconds(video_path: str, interval_seconds: int = 3) -> list[tuple[float, Image.Image]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_seconds = frame_count / fps if fps > 0 else 0

    frames: list[tuple[float, Image.Image]] = []
    timestamp = 0.0

    while True:
        if duration_seconds and timestamp > duration_seconds:
            break

        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = capture.read()
        if not success:
            break

        frames.append((timestamp, resize_frame(frame)))
        timestamp += interval_seconds

    capture.release()
    return frames


def describe_frame(client: Groq, image: Image.Image, timestamp: float) -> str:
    prompt = (
        "Describe this video frame in one short sentence. "
        "Focus on the main action, people, objects, and setting. "
        f"The frame was sampled at {timestamp:.0f} seconds."
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def summarize_scene(client: Groq, frame_notes: list[str]) -> str:
    joined_notes = "\n".join(f"- {note}" for note in frame_notes)
    prompt = (
        "You are given short descriptions of frames from one video.\n"
        "Write one concise scene description that summarizes what happens throughout the video.\n"
        "Keep it to 2-4 sentences and avoid listing every frame.\n\n"
        f"Frame notes:\n{joined_notes}"
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


st.title("Video Scene Describer")
st.write(
    "Upload one MP4 video, and the app will sample 1 frame every 3 seconds, "
    "analyze it, and generate a short scene description."
)

with st.sidebar:
    st.header("Settings")
    st.text_input("Groq vision model", value=VISION_MODEL, disabled=True)
    st.text_input(
        "Groq API key",
        key="groq_api_key",
        type="password",
        placeholder="gsk_...",
        help="Paste your Groq API key here if you do not want to use environment variables or Streamlit secrets.",
    )
    st.caption("This app is configured for Groq vision analysis with `meta-llama/llama-4-scout-17b-16e-instruct`.")

uploaded_video = st.file_uploader("Upload an MP4 video", type=["mp4"])

if uploaded_video:
    st.video(uploaded_video)

    if st.button("Analyze video", type="primary"):
        temp_path = None
        try:
            client = get_groq_client()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.getbuffer())
                temp_path = temp_file.name

            with st.spinner("Extracting frames..."):
                frames = extract_frames_every_n_seconds(temp_path, interval_seconds=3)

            if not frames:
                st.error("No frames could be extracted from the video.")
            else:
                st.success(f"Extracted {len(frames)} frames.")

                frame_notes = []
                progress = st.progress(0)
                status_text = st.empty()

                for index, (timestamp, frame) in enumerate(frames, start=1):
                    status_text.write(f"Analyzing frame {index} of {len(frames)} at {int(timestamp)}s...")
                    note = describe_frame(client, frame, timestamp)
                    frame_notes.append(f"{int(timestamp)}s: {note}")
                    progress.progress(index / len(frames))

                scene_description = summarize_scene(client, frame_notes)

                st.subheader("Scene Description")
                st.write(scene_description)

                with st.expander("Frame notes"):
                    for note in frame_notes:
                        st.write(note)

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
