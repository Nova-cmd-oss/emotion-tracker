import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import math
import cvlib as cv
from deepface import DeepFace

# --- Initialize Models (do this once globally) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# --- 1. Core Attributes Analysis Function ---
def analyze_core_attributes(image):
    """Analyzes age, gender, emotion, and race using DeepFace."""
    if image is None:
        return None, "Please upload an image."
    try:
        # DeepFace returns a list of dicts, one for each face. We'll use the first.
        analysis_results = DeepFace.analyze(
            img_path=image, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=True
        )
        first_face = analysis_results[0]
        
        # Draw bounding box
        region = first_face['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        image_with_box = image.copy()
        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Format results
        output_text = f"""
        ### Core Attributes
        ---
        - **Estimated Age:** {first_face.get('age', 'N/A')}
        - **Dominant Gender:** {first_face.get('dominant_gender', 'N/A').capitalize()}
        - **Dominant Emotion:** {first_face.get('dominant_emotion', 'N/A').capitalize()}
        - **Dominant Race:** {first_face.get('dominant_race', 'N/A').capitalize()}
        """
        return image_with_box, output_text
    except Exception as e:
        return image, f"**Error:** {str(e)}"



# --- Main Execution Block ---
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)



