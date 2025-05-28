import os
import tempfile
import requests

# Re-enabled heavy imports for current stage
import numpy as np
import librosa
from moviepy import VideoFileClip
import speech_recognition as sr
# import yt_dlp # This remains commented out

# Keep these imports commented out for now
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# import pickle
# import torch
# import torchaudio
# import transformers
# import soundfile # likely needed by librosa or moviepy, but let's see

# --- Dummy accent training/classification functions (still commented out) ---
# def create_sample_data():
#     pass
# def train_accent_classifier():
#     pass
# def extract_audio_features(audio_path):
#     pass

# --- Core video/audio processing functions (re-enabled) ---

# Download video (MODIFIED TO USE REQUESTS FOR DIRECT MP4 URL)
def download_video(video_url): # video_url here will be the direct MP4 URL
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "temp_video.mp4")

    # The hardcoded test video URL will be passed from process_video_url
    print(f"Attempting to download video from: {video_url} to {video_path}")

    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download video from {video_url}: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during video download: {e}")

    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        raise ValueError(f"Downloaded video file is invalid or corrupted. Size: {os.path.getsize(video_path)} bytes.")

    print(f"Successfully downloaded video to: {video_path}")
    return video_path, temp_dir


# Extract audio from video
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio

    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "temp_audio.wav")
    audio.write_audiofile(audio_path)

    return audio_path, temp_dir


# Transcribe speech to text
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "API unavailable"


# Classify accent (still commented out for now)
# def classify_accent(audio_path, model):
#     features = extract_audio_features(audio_path)
#     features = features.reshape(1, -1)  # Reshape for single sample

#     accent_map = {0: "American", 1: "British", 2: "Australian"}
#     pred = model.predict(features)[0]
#     proba = model.predict_proba(features)[0]
#     confidence = np.max(proba) * 100

#     return accent_map.get(pred, "Unknown"), confidence


# Main processing function (calls the above, still no accent classification)
def process_video_url(user_provided_url): # user_provided_url here is ignored
    # model = train_accent_classifier() # Still commented out

    # --- MODIFIED PART: Hardcode the test video URL here ---
    test_mp4_url = "http://techslides.com/demos/sample-videos/small.mp4"
    video_path, video_dir = download_video(test_mp4_url) # Call with the test URL
    # --- END MODIFIED PART ---

    try:
        audio_path, audio_dir = extract_audio(video_path)

        try:
            transcription = transcribe_audio(audio_path)

            results = {
                "accent": "Processing (no classification yet)",
                "confidence": "N/A",
                "transcription": transcription,
                "summary": f"Transcription: {transcription}"
            }
            return results
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(audio_dir):
                os.rmdir(audio_dir)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(video_dir):
            os.rmdir(video_dir)


# Flask web application setup
from flask import Flask, request, render_template, jsonify

basedir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basedir, 'templates')

# Debugging output
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {basedir}")
print(f"Templates directory: {template_dir}")
print(f"Templates exists: {os.path.exists(template_dir)}")
print(f"Templates contents: {os.listdir(template_dir)}")

# Initialize Flask with explicit template folder
app = Flask(__name__, template_folder=template_dir)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form.get('video_url')
        try:
            # This calls the process_video_url function with its newly enabled logic
            results = process_video_url(video_url)
            return render_template('results.html', results=results)
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)