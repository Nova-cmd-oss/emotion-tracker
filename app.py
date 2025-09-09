import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import math
import cvlib as cv
from deepface import DeepFace

# --- Initialize Models (do this once globally) ---
mp_face_mesh = mp.solutions.face_mesh
# Increased max_num_faces to handle more scenarios, but logic will process the first.
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
        analysis_results = DeepFace.analyze(
            img_path=image, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=True
        )
        first_face = analysis_results[0]
        
        region = first_face['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        image_with_box = image.copy()
        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
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
        return image, f"**Error:** Could not find a face or analyze the image. Details: {str(e)}"

# --- 2. Head Pose & Gaze Analysis Function ---
def analyze_pose_and_gaze(image):
    """Analyzes head pose, gaze, and blink detection using MediaPipe."""
    if image is None:
        return None, "Please upload an image."
    
    # **FIX:** Gradio provides an RGB image. MediaPipe also expects RGB.
    # The previous code was converting BGR to RGB, which was incorrect as the input was already RGB.
    # We just need to make a copy to work with.
    image_rgb = image.copy()
    results = face_mesh.process(image_rgb)
    
    annotated_image = image.copy()
    
    if not results.multi_face_landmarks:
        return image, "**Error:** Could not detect face landmarks. Please use a clearer, front-facing image."

    face_landmarks = results.multi_face_landmarks[0]
    img_h, img_w, _ = image.shape
    
    # --- Head Pose Estimation ---
    face_2d = []
    face_3d = []
    
    # Key landmarks for pose
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
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

    # Determine text for head pose
    if y_angle < -10:
        text_pose = "Looking Left"
    elif y_angle > 10:
        text_pose = "Looking Right"
    elif x_angle < -10:
        text_pose = "Looking Down"
    elif x_angle > 10:
        text_pose = "Looking Up"
    else:
        text_pose = "Forward"

    # --- Blink Detection (Eye Aspect Ratio) ---
    def get_ear(eye_points, landmarks):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
        A = np.linalg.norm(np.array([p2.x, p2.y]) - np.array([p6.x, p6.y]))
        B = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p5.x, p5.y]))
        C = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p4.x, p4.y]))
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373]
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153]
    EAR_THRESHOLD = 0.20

    left_ear = get_ear(LEFT_EYE_INDICES, face_landmarks.landmark)
    right_ear = get_ear(RIGHT_EYE_INDICES, face_landmarks.landmark)
    
    text_blink = "Eyes Open"
    if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
        text_blink = "Eyes Closed (Blink)"

    # --- Drawing and Results ---
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)
        
    output_text = f"""
    ### Pose & Gaze Analysis
    ---
    - **Head Pose:** {text_pose} (Rotation: {y_angle:.1f}°)
    - **Eye State:** {text_blink} (EAR: {((left_ear+right_ear)/2):.2f})
    
    *Gaze is estimated based on head rotation. EAR is the Eye Aspect Ratio.*
    """
    
    return annotated_image, output_text

# --- 3. Accessory Detection Function ---
def analyze_accessories(image):
    """Detects common accessories like eyeglasses using CVlib."""
    if image is None:
        return None, "Please upload an image."
    
    annotated_image = image.copy()
    # Using model='yolov4-tiny' can sometimes be faster
    bbox, labels, conf = cv.detect_common_objects(annotated_image, confidence=0.4, model='yolov4')
    
    # Filter for items that might be accessories
    allowed_accessories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'eyeglasses'] # Added eyeglasses explicitly
    detected_items = []
    
    for label, box in zip(labels, bbox):
        if label not in allowed_accessories:
            continue
        
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(annotated_image, (x, y), (w, h), (255, 0, 0), 2)
        cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        detected_items.append(label.capitalize())
        
    if not detected_items:
        output_text = "### Accessory Detection\n---\n- No common accessories detected. This works best on clear, high-resolution images."
    else:
        items_str = ", ".join(detected_items)
        output_text = f"### Accessory Detection\n---\n- **Detected:** {items_str}"

    return annotated_image, output_text


# --- Gradio Interface Setup ---
def create_gradio_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="Advanced Facial Analyzer") as interface:
        gr.Markdown(
            """
            # Advanced Facial Analyzer 🔬
            Choose a tab below to perform a specific type of facial analysis. Upload an image and click the button.
            **For best results, use clear, well-lit, front-facing images.**
            """
        )

        with gr.Tabs():
            # --- Tab 1: Core Attributes ---
            with gr.TabItem("Core Attributes (Age, Emotion...)"):
                with gr.Row():
                    with gr.Column():
                        input1 = gr.Image(type="numpy", label="Upload Image")
                        button1 = gr.Button("Analyze Core Attributes", variant="primary")
                    with gr.Column():
                        output1_img = gr.Image(label="Detected Face")
                        output1_txt = gr.Markdown(label="Results")
                button1.click(analyze_core_attributes, inputs=input1, outputs=[output1_img, output1_txt])
                gr.Examples(
                    ["assets/sample1.jpg", "assets_bkp/sample2.jpg"], input1, [output1_img, output1_txt], analyze_core_attributes, cache_examples=False
                )

            # --- Tab 2: Pose & Gaze ---
            with gr.TabItem("Head Pose & Gaze"):
                with gr.Row():
                    with gr.Column():
                        input2 = gr.Image(type="numpy", label="Upload Image")
                        button2 = gr.Button("Analyze Pose & Gaze", variant="primary")
                    with gr.Column():
                        output2_img = gr.Image(label="Facial Landmarks")
                        output2_txt = gr.Markdown(label="Results")
                button2.click(analyze_pose_and_gaze, inputs=input2, outputs=[output2_img, output2_txt])
                gr.Examples(
                    ["assets/sample3.jpg", "assets_bkp/sample4.jpg"], input2, [output2_img, output2_txt], analyze_pose_and_gaze, cache_examples=False
                )

            # --- Tab 3: Accessory Detection ---
            with gr.TabItem("Accessory Detection"):
                with gr.Row():
                    with gr.Column():
                        input3 = gr.Image(type="numpy", label="Upload Image")
                        button3 = gr.Button("Analyze Accessories", variant="primary")
                    with gr.Column():
                        output3_img = gr.Image(label="Detected Objects")
                        output3_txt = gr.Markdown(label="Results")
                button3.click(analyze_accessories, inputs=input3, outputs=[output3_img, output3_txt])
                gr.Examples(
                    ["assets_bkp/sample4.jpg", "assets/sample5.jpg"], input3, [output3_img, output3_txt], analyze_accessories, cache_examples=False
                )
        
        gr.Markdown(
            """
            ---
            *Powered by DeepFace, MediaPipe, and CVlib. Analysis is for educational purposes and may not be 100% accurate.*
            """
        )
    return interface

# --- Main Execution Block ---
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)

