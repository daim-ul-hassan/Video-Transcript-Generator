# Video Transcript Generator

A simple Streamlit app that accepts a single MP4 upload, extracts one frame every 3 seconds, sends each sampled frame to a Groq vision model, and generates one concise scene description for the whole video.

## Features

- MP4-only upload for a beginner-friendly workflow
- Automatic frame sampling every 3 seconds
- AI-based frame descriptions
- One clean final scene summary
- Simple Streamlit UI with progress feedback

## Requirements

- Python 3.10 or newer
- A Groq API key

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key:

```bash
set GROQ_API_KEY=your_api_key_here
```

Or add it to Streamlit secrets:

```toml
GROQ_API_KEY = "your_api_key_here"
```

You can also paste the key directly into the sidebar inside the app under "Groq API key".

## Run the app

```bash
streamlit run app.py
```

## How it works

1. Upload an MP4 video.
2. The app samples 1 frame every 3 seconds using OpenCV.
3. Each frame is analyzed by a Groq vision-capable model.
4. The frame-level notes are combined into one short scene description.

## Notes

- This version supports MP4 only.
- If the API key is missing, the app will show a clear error message.
- The app uses Groq's `meta-llama/llama-4-scout-17b-16e-instruct` vision model.
