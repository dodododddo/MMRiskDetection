import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def analyze_audio(input_audio_path):
    
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"

    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")
    elif not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is not None:
        scaler = joblib.load(scaler_filename)
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

        svm_classifier = joblib.load(model_filename)
        prediction = svm_classifier.predict(mfcc_features_scaled)

        if prediction[0] == 0:
            if(file.startswith('p') or file.startswith('SS')):
                print('A')
                return True
            else:
                return False
        else:
            if(file.startswith('F')):
                print('B')
                return True
            else:
                return False
    else:
        print("Error: Unable to process the input audio.")

total = 0
success = 0

for root, dirs, files in os.walk('../Audio_corpus/C6'):
    for file in files:
        print(file)
        total = total + 1
        if(analyze_audio(f'../Audio_corpus/C6/{file}')):
            success = success + 1

print(total)
print(success)

print(float(success) / float(total))