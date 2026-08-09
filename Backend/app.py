import os
import numpy as np
import librosa
import opensmile
import joblib
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Maximum file size (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Load trained artifacts
model = joblib.load("xgboost_parkinson_model.pkl")
scaler = joblib.load("scaler.pkl")
indices = np.load("feature_indices.npy")

# Initialize OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def compute_cpp(signal, sr):
    spectrum = np.fft.fft(signal)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real
    return np.max(cepstrum)


def extract_features(audio_path):

    # Extract OpenSMILE features
    features = smile.process_file(audio_path).reset_index(drop=True)

    # Load audio for CPP
    y, sr = librosa.load(audio_path, sr=44100)

    cpp = compute_cpp(y, sr)

    # Add CPP feature
    features["cpp"] = cpp

    X = features.values

    # Scale features
    X_scaled = scaler.transform(X)

    # Select features
    X_selected = X_scaled[:, indices]

    return X_selected


@app.route("/predict", methods=["POST"])
def predict():

    # Test Case 1 — No file uploaded
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    # Test Case 2 — Empty filename
    if audio_file.filename == "":
        return jsonify({"error": "Audio file is empty"}), 400

    # Test Case 3 — Invalid format
    if not audio_file.filename.lower().endswith(".wav"):
        return jsonify({"error": "Invalid audio format. Please upload a WAV file"}), 400

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_file.save(temp.name)
        audio_path = temp.name

    try:

        # Load audio
        y, sr = librosa.load(audio_path, sr=44100)

        # Test Case 4 — Empty audio signal
        if len(y) == 0:
            return jsonify({"error": "Audio file contains no sound"}), 400

        # Test Case 5 — Audio too short
        duration = librosa.get_duration(y=y, sr=sr)

        if duration < 1:
            return jsonify({
                "error": "Audio too short for analysis. Please upload at least 1 second audio"
            }), 400

        # Extract features
        X_selected = extract_features(audio_path)

        # Prediction
        prob = model.predict_proba(X_selected)[0][1]
        pred = model.predict(X_selected)[0]

        result = "Parkinson Detected" if pred == 1 else "Healthy"

        return jsonify({
            "prediction": int(pred),
            "probability": float(prob),
            "result": result
        })

    # Test Case 6 — Corrupted audio
    except Exception as e:
        return jsonify({
            "error": "Unable to process audio file",
            "details": str(e)
        }), 500

    finally:
        # Delete temp file
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    app.run(debug=True)