# Advanced Facial Analyzer  

A multi-featured **facial analysis web application** built with **Python** and **Gradio**. This tool allows users to analyze facial attributes using different models and methods, powered by **DeepFace**, **MediaPipe**, and **CVlib**.  

The application provides a **tabbed interface** with three main analysis modes:  
1. **Core Attributes** → Age, Gender, Emotion, and Race detection (via DeepFace).  
2. **Head Pose & Gaze** → Facial mesh visualization, head direction estimation, and blink detection (via MediaPipe).  
3. **Accessory Detection** → Object detection for common facial accessories such as eyeglasses (via CVlib).  

---

---

## ⚙️ Setup & Installation  

### 1. Prerequisites  
- **Python 3.8+**  
- A working internet connection (first run will download pre-trained models).  

### 2. Clone and Set Up Project  
```bash
# Clone or create project folder
cd path/to/projects
mkdir emotion_age_detector && cd emotion_age_detector

### 3. Create & Activate Virtual Environment
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

🚀 Application Features
🔹 Core Attributes (DeepFace)

Age estimation

Gender detection

Emotion recognition

Race classification

🔹 Head Pose & Gaze (MediaPipe)

Facial mesh visualization

Head direction estimation: Up, Down, Left, Right

Eye blink detection (eyes open/closed)

🔹 Accessory Detection (CVlib)

Object detection on faces

Recognition of common items like eyeglasses

🛠️ Tech Stack

Python
Gradio
DeepFace
MediaPipe
CVlib






