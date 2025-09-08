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




# --- Main Execution Block ---
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)



