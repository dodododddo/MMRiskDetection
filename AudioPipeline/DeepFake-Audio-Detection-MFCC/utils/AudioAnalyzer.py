import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

class AudioAnalyzer:
    def __init__(self, model_filename="svm_model.pkl", scaler_filename="scaler.pkl"):
        self.model_filename = model_filename
        self.scaler_filename = scaler_filename
        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _load_model(self):
        if os.path.exists(self.model_filename):
            return joblib.load(self.model_filename)
        else:
            raise FileNotFoundError(f"Model file {self.model_filename} not found.")
        
    def _load_scaler(self):
        if os.path.exists(self.scaler_filename):
            return joblib.load(self.scaler_filename)
        else:
            raise FileNotFoundError(f"Scaler file {self.scaler_filename} not found.")

    def extract_mfcc_features(self, audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
        try:
            audio_data, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None

        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfccs.T, axis=0)

    def analyze_audio(self, input_audio_path):
        if not os.path.exists(input_audio_path):
            return "Error: The specified file does not exist."
        elif not input_audio_path.lower().endswith(".wav"):
            return "Error: The specified file is not a .wav file."

        mfcc_features = self.extract_mfcc_features(input_audio_path)
        if mfcc_features is not None:
            mfcc_features_scaled = self.scaler.transform(mfcc_features.reshape(1, -1))
            print(self.model)
            prediction = self.model.predict(mfcc_features_scaled)
            print(prediction)

            if prediction[0] == 0:
                return False
            else:
                return True
        else:
            return "Error: Unable to process the input audio."
