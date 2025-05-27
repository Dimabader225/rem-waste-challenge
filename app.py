import os
import tempfile
import requests
import numpy as np
import librosa
from moviepy import VideoFileClip
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import speech_recognition as sr

import yt_dlp
import tempfile
import os

# Sample accent data (in a real app, we'd use a proper dataset)
# Format: (audio_features, accent_label)
# 0: American, 1: British, 2: Australian
def create_sample_data():
    # This is just a placeholder - real implementation would use proper training data
    np.random.seed(42)
    X = np.random.rand(100, 20)  # 100 samples, 20 features each
    y = np.random.randint(0, 3, 100)  # Random labels
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
    y, sr = librosa.load(audio_path, sr=None)

    features = []
    # MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    features.extend(np.mean(mfcc, axis=1))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))

    return np.array(features[:20])  # Limit to 20 features for our dummy model






def download_video(video_url):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "temp_video.mp4")

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': video_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }]
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        raise ValueError(f"yt_dlp failed to download and merge video: {e}")

    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        raise ValueError("Downloaded video file is invalid or corrupted.")

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


# Main processing function
def process_video_url(video_url):
    # Initialize model (in production, load a pre-trained model)
    model = train_accent_classifier()

    # Download video
    video_path, video_dir = download_video(video_url)

    try:
        # Extract audio
        audio_path, audio_dir = extract_audio(video_path)

        try:
            # Analyze audio
            transcription = transcribe_audio(audio_path)
            accent, confidence = classify_accent(audio_path, model)

            # Prepare results
            results = {
                "accent": accent,
                "confidence": f"{confidence:.1f}%",
                "transcription": transcription,
                "summary": f"The speaker has a {accent} accent with {confidence:.1f}% confidence."
            }

            return results
        finally:
            # Clean up audio files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(audio_dir):
                os.rmdir(audio_dir)
    finally:
        # Clean up video files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(video_dir):
            os.rmdir(video_dir)


# Flask web application
import os
from flask import Flask, request, render_template, jsonify

basedir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basedir, 'templates')

# Keep these print statements for debugging, they don't harm
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {basedir}")
print(f"Templates directory: {template_dir}")
print(f"Templates exists: {os.path.exists(template_dir)}")
print(f"Templates contents: {os.listdir(template_dir)}")

app = Flask(__name__, template_folder=template_dir)

@app.route('/', methods=['GET', 'POST'])
def index():
    # --- TEMPORARY DEBUGGING CODE: This line will run instead of your original logic ---
    return "Hello from Render! Your basic Flask app is running."
    # --- END TEMPORARY DEBUGGING CODE ---

    # The original code for handling POST requests will be commented out for now
    # if request.method == 'POST':
    #     video_url = request.form.get('video_url')
    #     try:
    #         results = process_video_url(video_url)
    #         return render_template('results.html', results=results)
    #     except Exception as e:
    #         return render_template('error.html', error=str(e))
    # # The original code for GET requests will also be commented out
    # return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)