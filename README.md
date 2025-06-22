# 🛰️ Drone Detection WebRTC Backend

This is a Python-based backend server using **FastAPI**, **WebRTC (aiortc)**, and **YOLOv8** for real-time video processing. Currently, it performs **face detection** using a model from Hugging Face, but the detection logic is easily replaceable with a any other object detection model later.

---

## 🚀 Features

- 📹 Accepts video stream from browser via WebRTC
- 🧠 Processes video frames using YOLOv8
- 🎯 Detects objects (for now we are using face detection model)
- 🪄 Returns annotated video stream to frontend in real-time
- ⚡ Async & frame-skipping logic for smooth performance

---

## 📦 Requirements

- Python 3.8+
- Frontend (React/Vite/etc.) that sends WebRTC offer

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/your-org/drone-detection-backend.git
cd drone-detection-backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
---

## 📄 Model Attribution

This project uses the YOLOv8 face detection model from:

- **Model Name:** YOLOv8-Face-Detection  
- **Author:** [@arnabdhar](https://huggingface.co/arnabdhar)  
- **Source:** [Hugging Face](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)  
- **License:** MIT License
