import os
import tempfile
import requests

# Re-enabled heavy imports for current stage
import numpy as np
import librosa
from moviepy import VideoFileClip
import speech_recognition as sr
# import yt_dlp # This remains commented out

# Re-enabled imports for accent classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
# import torch # Still commented out
# import torchaudio # Still commented out
# import transformers # Still commented out
# import soundfile # Still commented out

# --- Accent training/classification functions (ALL UNCOMMENTED) ---
# Sample accent data (in a real app, we'd use a proper dataset)
# Format: (audio_features, accent_label)
# 0: American, 1: British, 2: Australian
def create_sample_data():
    # This is just a placeholder - real implementation would use proper training data
    np.random.seed(42)
    X = np.random.rand(100, 20)  # 100 samples, 20 features each
    y = np.random.randint(0, 3, 100)  # Random labels (0: American, 1: British, 2: Australian)
    return X, y

# Train a simple classifier (in production, we'd use a pre-trained model)
def train_accent_classifier():
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100)
    )
    model.fit(X_train, y_train)
    return model

# Extract audio features using librosa
def extract_audio_features(audio_path):
    y, sr_audio = librosa.load(audio_path, sr=None) # Using sr_audio to avoid conflict with global sr (SpeechRecognition)

    features = []
    # MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_audio)
    features.extend(np.mean(mfcc, axis=1))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr_audio)
    features.extend(np.mean(contrast, axis=1))

    return np.array(features[:20]) # Limit to 20 features for our dummy model

# Classify accent
def classify_accent(audio_path, model):
    features = extract_audio_features(audio_path)
    features = features.reshape(1, -1)  # Reshape for single sample

    # Predict accent
    accent_map = {0: "American", 1: "British", 2: "Australian"}
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = np.max(proba) * 100

    return accent_map.get(pred, "Unknown"), confidence

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

    print(f"Successfully downloaded video to: {video_path}. File size: {os.path.getsize(video_path)} bytes.")
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


# Main processing function (calls the above, now with accent classification)
def process_video_url(user_provided_url): # user_provided_url here is ignored
    model = train_accent_classifier() # UNCOMMENTED

    # Hardcode the test video URL here
    test_mp4_url = "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
    video_path, video_dir = download_video(test_mp4_url)

    try:
        audio_path, audio_dir = extract_audio(video_path)

        try:
            transcription = transcribe_audio(audio_path)
            accent, confidence = classify_accent(audio_path, model) # UNCOMMENTED

            results = {
                "accent": accent, # RESTORED
                "confidence": f"{confidence:.1f}%", # RESTORED
                "transcription": transcription,
                "summary": f"The speaker has a {accent} accent with {confidence:.1f}% confidence. Transcription: {transcription}" # RESTORED
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