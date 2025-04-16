from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf
import joblib
import os

# Ensure Flask finds templates
app = Flask(__name__, template_folder="templates")

# Load trained model and preprocessing tools
model = tf.keras.models.load_model("fraud_detection_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

# Function to extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1)

# Process audio file
def preprocess_audio(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])  # Normalize
    return np.array(features).reshape(1, 40, 1)

# Route for frontend
@app.route('/')
def home():
    return render_template("index.html")

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "temp_audio.wav"
    file.save(file_path)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not saved correctly"}), 500

    features = preprocess_audio(file_path)
    prediction = model.predict(features)

    result = "Fraud Call" if prediction[0] > 0.5 else "Real Call"
    
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)





