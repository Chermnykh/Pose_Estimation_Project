import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp
import numpy as np
import threading
import base64
import time
lock = threading.Lock()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = round(np.abs(radians*180.0/np.pi))
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Counter variables
counter = 0 
stage = None
advice = None
exercise = None
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

st.title("SmartFit")

video_file = open('curl.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)

def callback(frame):
    global counter
    global stage
    global exercise
    global advice
    image = frame.to_ndarray(format="bgr24")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    with lock:
        results = pose.process(image)
    if results.pose_landmarks is None:
        return frame

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    landmarks = results.pose_landmarks.landmark
    
    # Get coordinates (right)
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # Get coordinates (left)
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # Calculate angle
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # Visualize angle
    cv2.putText(image, str(right_elbow_angle), 
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                        )
    
    cv2.putText(image, str(left_elbow_angle), 
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                        )
    
    # Curl counter logic
    if exercise == None:
        exercise = "Right curl"
    if exercise == "Right curl":
        if right_elbow_angle > 160:
            stage = "down"
        if right_elbow_angle < 30 and stage =='down' and (right_elbow[0] - right_shoulder[0]) <= 0.15:
            stage="up"
            counter +=1
        if stage == "up" and right_elbow_angle > 140 and exercise == "Right curl":
            exercise = "Left curl"
    if exercise == "Left curl":
        if left_elbow_angle > 160:
            stage = "down"
        if left_elbow_angle < 30 and stage =='down' and (left_elbow[0] - left_shoulder[0]) <= 0.15:
            stage="up"
            counter +=1
        if stage == "up" and left_elbow_angle > 140 and exercise == "Left curl":
            exercise = "Right curl"

    # Advice
    if 60 < right_elbow_angle < 120 and stage == "down":
        advice = "Pull your arm"
    elif 60 < right_elbow_angle < 120 and stage == "up":
        advice = "Straighten your arm" 
    elif right_ankle[0] > right_shoulder[0]  and left_ankle[0] < left_shoulder[0]:
        advice = "Legs wider"
    else:
        advice = None
    
    # Setup status box
    cv2.rectangle(image, (0,500), (800,425), (135, 206, 235), -1)
    # Advice
    cv2.putText(image, advice, 
                (10,470), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 2, cv2.LINE_AA)

    # Render counter
    # Setup status box
    cv2.rectangle(image, (0,0), (320,73), (135, 206, 235), -1)
    # Exercise name
    cv2.putText(image, 'Exercise', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, exercise, 
                (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


    # Setup status box
    cv2.rectangle(image, (0,75), (320,148), (135, 206, 235), -1)
    
    # Rep data
    cv2.putText(image, 'REPS', (15,87), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    
    cv2.putText(image, str(counter), 
                (10,135), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    # Stage data
    cv2.putText(image, 'STAGE', (130,87), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, stage, 
                (120,135), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(232, 165, 22), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(128, 231, 69), thickness=2, circle_radius=2) 
                                )            
    return av.VideoFrame.from_ndarray(image, format="bgr24")

st.text("Lift the dumbbells alternately!")

webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
