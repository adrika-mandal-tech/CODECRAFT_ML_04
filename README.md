# 🤟 American Sign Language (ASL) Detector using Hand Gestures

This project is a real-time **American Sign Language (ASL) alphabet detector** that recognizes **all 26 letters (A–Z)** using hand gestures captured via webcam.

It combines the power of **Computer Vision** and **Deep Learning** to interpret gestures based on ASL rules — making communication more inclusive and accessible.

---

## 🧠 What It Does

📸 **Captures hand gestures** in real-time using your webcam  
🧾 **Classifies them** into one of the 26 English alphabets (A–Z)  
🖐️ Follows standard **American Sign Language** rules  
📊 Uses a **trained ML model** for accurate alphabet prediction

---

## 🧪 Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| 🔍 OpenCV     | Capturing and processing real-time video feed |
| 🤖 TensorFlow | Building and loading the trained ML model |
| 🖐️ MediaPipe  | Real-time hand tracking and landmark detection |

---



## 🗂️ Project Structure

```
ASL_Detector/
│
├── Model/                # Trained .tflite or .h5 model
│
├── dataset/              # Folder containing training images (A–Z)
│
├── screenshots/          # Visuals and sample output images
│
├── asl_detector.py       # Main Python script for running real-time ASL detection
│
├── requirements.txt      # List of Python dependencies
│
└── README.md             # Project documentation
```
