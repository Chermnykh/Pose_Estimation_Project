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
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Counter variables
counter = 0 
stage = None
exercise = None
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    import streamlit as st
    import base64

    def intro():
        st.title("SmartFit")

        st.subheader("Exercise Plan: \n * Squats \n * Curl")

        audio_base64 = base64.b64encode(open("knees.wav", "rb").read()).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
        st.markdown(audio_tag, unsafe_allow_html=True)

    def squats():
        st.title("SmartFit")
        def callback(frame):
            global counter
            global stage
            global exercise

            exercise = "Squats"
            
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
            
            # Get coordinates
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            vertical = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 0]

            # Calculate angle
            knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            body_angle = calculate_angle(right_shoulder, right_hip, vertical)

            # Visualize angle
            cv2.putText(image, str(knee_angle), 
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            # Squats counter logic
            if knee_angle > 140:
                stage = "up"
            if knee_angle < 90 and stage == "up" and right_knee[0] < right_foot_index[0] and 20 <= body_angle <= 45:
                stage = "down"
                counter += 1

            # Advice
            # if right_knee[0] > right_foot_index[0]:
            #     audio_base64 = base64.b64encode(open("knees.wav", "rb").read()).decode('utf-8')
            #     audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
            #     st.markdown(audio_tag, unsafe_allow_html=True)

            # Render counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (161, 136, 189), -1)
            # Exercise name
            cv2.putText(image, 'Exercise', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, exercise, 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Setup status box
            cv2.rectangle(image, (0,75), (225,148), (161, 136, 189), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,87), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(counter), 
                        (10,135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,87), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (70,135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )            
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        webrtc_streamer(
            key="example",
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    def curl():
        st.title("SmartFit")
        def callback(frame):
            global counter
            global stage
            global exercise

            exercise = 'Curl'

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
            
            # Get coordinates
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
            # Visualize angle
            cv2.putText(image, str(right_elbow_angle), 
                            tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Curl counter logic
            if right_elbow_angle > 160:
                stage = "down"
                # and right_shoulder[0] <= right_ankle[0]
            if right_elbow_angle < 30 and stage =='down' and (right_elbow[0] - right_shoulder[0]) <= 0.15:
                stage="up"
                counter +=1

            # Render counter
            # Setup status box
            cv2.rectangle(image, (0,0), (245,73), (161, 136, 189), -1)
            # Exercise name
            cv2.putText(image, 'Exercise', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, exercise, 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


            # Setup status box
            cv2.rectangle(image, (0,75), (245,148), (161, 136, 189), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,87), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(counter), 
                        (10,135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (115,87), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (90,135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )            
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        webrtc_streamer(
            key="example",
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )



    page_names_to_funcs = {
        "—": intro,
        "Squats": squats,
        "Curl": curl
    }

    demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
