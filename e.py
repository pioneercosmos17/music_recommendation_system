import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

# Load pre-trained model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe holistic model
holistic = mp.solutions.holistic
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Streamlit header
st.header("Emotion Based Music Recommender")

# Check if 'run' is in session state, initialize if not
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load emotion from file or set to empty string if not found
try:
    emotion = np.load("emotion.npy")[0]
except (FileNotFoundError, IndexError):
    emotion = ""

# Set 'run' session state based on emotion status
if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# EmotionProcessor class for emotion detection
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Flip frame horizontally
        frm = cv2.flip(frm, 1)

        # Process frame with MediaPipe holistic model
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        # Initialize empty list for landmark coordinates
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            # Convert list to numpy array and reshape
            lst = np.array(lst).reshape(1, -1)

            # Predict emotion label using pre-trained model
            pred = label[np.argmax(model.predict(lst))]

            # Display predicted emotion on frame
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            # Save predicted emotion to file
            try:
                np.save("emotion.npy", np.array([pred]))
            except Exception as e:
                st.error(f"Error saving emotion: {e}")

        # Draw face landmarks on frame
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,
                                                                         circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))

        # Return processed frame
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Input fields for language and singer
lang = st.text_input("Language")
singer = st.text_input("Singer")

# Start webcam stream if language, singer, and 'run' session state are valid
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

# Button to recommend songs based on detected emotion
btn = st.button("Recommend me songs")

# Handle button click event
if btn:
    # Check if emotion is detected
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        # Open YouTube search with language, emotion, and singer keywords
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")

        # Clear emotion and update 'run' session state
        try:
            np.save("emotion.npy", np.array([""]))
        except Exception as e:
            st.error(f"Error clearing emotion: {e}")
        st.session_state["run"] = "false"
