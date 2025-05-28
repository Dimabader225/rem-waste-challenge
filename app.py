import os
import tempfile
import requests

# Re-enabled heavy imports for current stage
import numpy as np
import librosa
from moviepy import VideoFileClip # Keep for extract_audio if you re-enable it later
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

# --- Core video/audio processing functions (modified for MP3 test) ---

# This function is not called in the MP3 test, but kept for future use
def download_video(video_url):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "temp_video.mp4")

    print(f"Attempting to download video from: {video_url} to {video_path}")

    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

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


# This function is not called in the MP3 test, but kept for future use
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio

    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "temp_audio.wav")
    audio.write_audiofile(audio_path)

    return audio_path, temp_dir


# Transcribe speech to text (same as before)
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


# --- Global Model Training (RUNS ONCE AT APP STARTUP) ---
# This is the correct way to train a model in a web service
# It will run when Gunicorn starts a worker process
try:
    GLOBAL_ACCENT_MODEL = train_accent_classifier()
    print("Accent classifier model trained successfully on startup.")
except Exception as e:
    GLOBAL_ACCENT_MODEL = None # Set to None if training fails
    print(f"ERROR: Could not train accent classifier on startup: {e}")


# Main processing function (MODIFIED FOR TINY MP3 TEST)
def process_video_url(user_provided_url): # user_provided_url is still ignored for this test
    # Use the globally trained model
    if GLOBAL_ACCENT_MODEL is None:
        # If model failed to train on startup, raise an error or return a specific message
        raise ValueError("Accent classifier model is not available due to startup error.")
    model = GLOBAL_ACCENT_MODEL

    # --- NEW: Download a tiny MP3 directly ---
    test_mp3_url = "https://www.mfiles.co.uk/mp3-midi/test-mp3.mp3" # A 10KB MP3 file
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "test_audio.mp3") # Save as MP3

    print(f"Attempting to download test MP3 from: {test_mp3_url} to {audio_path}")
    try:
        response = requests.get(test_mp3_url, stream=True)
        response.raise_for_status()
        with open(audio_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded test MP3 to: {audio_path}. File size: {os.path.getsize(audio_path)} bytes.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download test MP3 from {test_mp3_url}: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during MP3 download: {e}")

    audio_dir = temp_dir # Use the same temp_dir for cleanup

    try:
        transcription = transcribe_audio(audio_path)
        accent, confidence = classify_accent(audio_path, model)

        results = {
            "accent": accent,
            "confidence": f"{confidence:.1f}%",
            "transcription": transcription,
            "summary": f"The speaker has a {accent} accent with {confidence:.1f}% confidence. Transcription: {transcription}"
        }
        return results
    finally:
        # Clean up the downloaded MP3 and its temp directory
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(audio_dir):
            os.rmdir(audio_dir)
        # No video files to clean up in this specific test
        pass


# Flask web application setup (unchanged)
from flask import Flask, request, render_template, jsonify

basedir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basedir, 'templates')

# Debugging output for startup
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
        video_url = request.form.get('video_url') # Still gets URL, but it's ignored for this test
        try:
            results = process_video_url(video_url)
            return render_template('results.html', results=results)
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)