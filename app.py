import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp
import numpy as np
import threading

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
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def callback(frame):
    global counter
    global stage
    image = frame.to_ndarray(format="bgr24")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    with lock:
        results = pose.process(image)
    if results.pose_landmarks is None:
        print("return to camera")
        return frame
    print(results)

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
    
    # Calculate angle
    knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    body_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    
    # Visualize angle
    cv2.putText(image, str(knee_angle), 
                    tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
    
    # Curl counter logic
    if knee_angle > 140:
        stage = "up"
    if knee_angle < 90 and stage == "up" and right_knee[0] < right_foot_index[0]:
        stage = "down"
        counter += 1
        print(counter)

    # Render counter
    # Setup status box
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
    
    # Rep data
    cv2.putText(image, 'REPS', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), 
                (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    # Stage data
    cv2.putText(image, 'STAGE', (65,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, stage, 
                (60,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )            
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=callback)


#st.title("My first Streamlit app")
#st.write("Hello, world")

#webrtc_streamer(key="example")



# import cv2
# import mediapipe as mp
# import numpy as np

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# def calculate_angle(a,b,c):
#     a = np.array(a) # First
#     b = np.array(b) # Mid
#     c = np.array(c) # End
    
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians*180.0/np.pi)
    
#     if angle >180.0:
#         angle = 360-angle
        
#     return angle 


# cap = cv2.VideoCapture(0)

# # Counter variables
# counter = 0 
# stage = None

# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
      
#         # Make detection
#         results = pose.process(image)
    
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         # Extract landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark
            
#             # Get coordinates
#             right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#             right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#             right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
#             right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            
#             # Calculate angle
#             angle = calculate_angle(right_hip, right_knee, right_ankle)
            
#             # Visualize angle
#             cv2.putText(image, str(angle), 
#                            tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                                 )
            
#             # Counter logic
#             if angle > 140:
#                 stage = "up"
#             if angle < 90 and stage == "up" and right_knee[0] < right_foot_index[0]:
#                 stage = "down"
#                 counter += 1
#                 print(counter)
                       
#         except:
#             pass
        
#         # Render counter
#         # Setup status box
#         cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
#         # Rep data
#         cv2.putText(image, 'REPS', (15,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#         cv2.putText(image, str(counter), 
#                     (10,60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
#         # Stage data
#         cv2.putText(image, 'STAGE', (65,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#         cv2.putText(image, stage, 
#                     (60,60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
#                                  )               
        
#         cv2.imshow('Mediapipe Feed', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()




    

# class VideoTransformer(VideoTransformerBase):
#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

#         return img


# webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)