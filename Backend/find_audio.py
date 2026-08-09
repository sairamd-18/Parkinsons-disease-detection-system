import numpy as np
import librosa
import opensmile
import joblib

model = joblib.load("xgboost_parkinson_model.pkl")
scaler = joblib.load("scaler.pkl")
indices = np.load("feature_indices.npy")

def compute_cpp(signal, sr):
    spectrum = np.fft.fft(signal)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real
    return np.max(cepstrum)


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def predict(audio_path):

    features = smile.process_file(audio_path).reset_index(drop=True)

    y, sr = librosa.load(audio_path, sr=44100)
    cpp = compute_cpp(y, sr)

    features["cpp"] = cpp

    X = features.values

    X_scaled = scaler.transform(X)

    X_selected = X_scaled[:, indices]

    prob = model.predict_proba(X_selected)[0][1]
    pred = model.predict(X_selected)[0]

    print("\nParkinson Probability:", prob)

    if pred == 1:
        print("Prediction: Parkinson Detected")
    else:
        print("Prediction: Healthy")


predict("test_audio.wav")
