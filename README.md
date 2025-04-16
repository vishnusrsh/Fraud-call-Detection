# 📞 Fraud Call Detection using Deep Learning

This is a deep learning-based real-time fraud/spam call detection system. The model classifies phone calls as **Fraud** or **Not Fraud** using audio data, and it's deployed using Flask for easy access via a web interface.

---

## 🔍 Overview

With the increasing number of scam calls, this project aims to detect fraudulent phone calls using machine learning techniques. The app allows users to test audio samples and receive instant predictions based on a trained deep learning model.

---

## 📂 Dataset

The dataset used is from Kaggle:

🔗 [Phone Call Spam Detection - Kaggle](https://www.kaggle.com/datasets/vemisettipavanbalaji/phone-call-spam-detection)

It contains audio samples categorized into:
- **Fraudulent Calls**
- **Real/Legitimate Calls**

---

## 🧠 Model

- Preprocessing: Mel spectrograms / MFCC features extraction
- Model: Deep Learning (e.g., CNN / LSTM)
- Frameworks: TensorFlow / Keras / Librosa for audio processing

---

## 🚀 How to Run

### 1. Clone the repository


git clone https://github.com/vishnusrsh/Fraud-call-Detection.git
cd Fraud-call-Detection


### 2. Install requirements
pip install -r requirements.txt

###3. Run the Flask App
python app.py

### Requirements
Make sure you have the following libraries installed:

flask
tensorflow
librosa
numpy
scikit-learn
matplotlib
pandas
soundfile
