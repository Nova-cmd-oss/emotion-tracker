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

# --- 2. Head Pose & Gaze Analysis Function ---
def analyze_pose_and_gaze(image):
    """Analyzes head pose, gaze, and blink detection using MediaPipe."""
    if image is None:
        return None, "Please upload an image."
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    annotated_image = image.copy()
    
    if not results.multi_face_landmarks:
        return image, "Could not detect face landmarks. Try a clearer image."

    face_landmarks = results.multi_face_landmarks[0]
    img_h, img_w, _ = image.shape
    
    # --- Head Pose Estimation ---
    face_2d = []
    face_3d = []
    
    # Get 2D landmarks for pose
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]: # Key landmarks for pose
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    x_angle = angles[0] * 360
    y_angle = angles[1] * 360
    z_angle = angles[2] * 360



# --- Main Execution Block ---
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)



