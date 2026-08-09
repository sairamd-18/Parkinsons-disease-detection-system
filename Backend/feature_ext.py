# clean_voice_feature_extraction.py

import os
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
import parselmouth
from scipy.signal import lfilter


# ==============================
# PRE-EMPHASIS FILTER
# ==============================
def pre_emphasis(signal, coeff=0.97):
    return lfilter([1, -coeff], [1], signal)


# ==============================
# MAIN FEATURE EXTRACTION
# ==============================
def extract_features(file_path):

    try:
        # 1️⃣ Load and Resample
        signal, sr = librosa.load(file_path, sr=16000)

        # 2️⃣ Noise Reduction
        signal = nr.reduce_noise(y=signal, sr=sr)

        # 3️⃣ Trim Silence
        signal, _ = librosa.effects.trim(signal, top_db=20)

        # Avoid empty signals
        if len(signal) < 1000:
            return None

        # 4️⃣ Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-6)

        # 5️⃣ Pre-emphasis
        signal = pre_emphasis(signal)

        features = {}

        # =============================
        # MFCC + Delta
        # =============================
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)

        for i in range(20):
            features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
            features[f"mfcc_{i}_std"] = np.std(mfcc[i])
            features[f"delta_{i}_mean"] = np.mean(mfcc_delta[i])
            features[f"delta_{i}_std"] = np.std(mfcc_delta[i])

        # =============================
        # Spectral Features
        # =============================
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(signal)

        features["spectral_centroid_mean"] = np.mean(spectral_centroid)
        features["spectral_centroid_std"] = np.std(spectral_centroid)

        features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
        features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)

        features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
        features["spectral_rolloff_std"] = np.std(spectral_rolloff)

        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)

        # =============================
        # PRAAT FEATURES (IMPORTANT FOR PD)
        # =============================
        snd = parselmouth.Sound(file_path)

        # Pitch
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values != 0]

        if len(pitch_values) > 0:
            features["pitch_mean"] = np.mean(pitch_values)
            features["pitch_std"] = np.std(pitch_values)
        else:
            features["pitch_mean"] = 0
            features["pitch_std"] = 0

        # Jitter & Shimmer
        point_process = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 500
        )

        features["jitter"] = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )

        features["shimmer"] = parselmouth.praat.call(
            [snd, point_process],
            "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        # Harmonic-to-Noise Ratio
        harmonicity = parselmouth.praat.call(
            snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
        )

        features["hnr"] = parselmouth.praat.call(
            harmonicity, "Get mean", 0, 0
        )

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ==============================
# PROCESS FOLDER
# ==============================
def process_dataset(audio_folder):

    dataset = []

    for root, _, files in os.walk(audio_folder):
        for file in files:
            if file.lower().endswith(".wav"):

                file_path = os.path.join(root, file)
                print("Processing:", file)

                features = extract_features(file_path)

                if features is not None:
                    features["label"] = 1 if "PD" in file else 0
                    features["AudioPath"] = file_path
                    dataset.append(features)

    df = pd.DataFrame(dataset)
    df.to_csv("processed_audio_features.csv", index=False)

    print("\nSaved: processed_audio_features.csv")
    print("Total samples processed:", len(df))
    print("Features per sample:", df.shape[1] - 2)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    audio_folder = "../data/audios"
    process_dataset(audio_folder)