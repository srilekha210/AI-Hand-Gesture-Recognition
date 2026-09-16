import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="✋",
    layout="centered"
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("✋ AI Hand Gesture Recognition")

st.write(
    "Use your webcam to capture your hand and recognize "
    "basic hand gestures using MediaPipe."
)

st.info(
    "Supported gestures: Open Hand, Fist, Thumbs Up, Victory / Peace"
)

# -----------------------------------
# MEDIAPIPE SETUP
# -----------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -----------------------------------
# GESTURE RECOGNITION
# -----------------------------------

def recognize_gesture(landmarks):

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    index_tip = landmarks[8]
    index_pip = landmarks[6]

    middle_tip = landmarks[12]
    middle_pip = landmarks[10]

    ring_tip = landmarks[16]
    ring_pip = landmarks[14]

    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]

    # Finger states

    index_open = index_tip.y < index_pip.y
    middle_open = middle_tip.y < middle_pip.y
    ring_open = ring_tip.y < ring_pip.y
    pinky_open = pinky_tip.y < pinky_pip.y

    # Thumb

    thumb_up = thumb_tip.y < thumb_ip.y

    # --------------------------------
    # THUMBS UP
    # --------------------------------

    if (
        thumb_up
        and not index_open
        and not middle_open
        and not ring_open
        and not pinky_open
    ):
        return "👍 Thumbs Up"

    # --------------------------------
    # VICTORY
    # --------------------------------

    if (
        index_open
        and middle_open
        and not ring_open
        and not pinky_open
    ):
        return "✌️ Victory / Peace"

    # --------------------------------
    # OPEN HAND
    # --------------------------------

    if (
        index_open
        and middle_open
        and ring_open
        and pinky_open
    ):
        return "✋ Open Hand"

    # --------------------------------
    # FIST
    # --------------------------------

    if (
        not index_open
        and not middle_open
        and not ring_open
        and not pinky_open
        and not thumb_up
    ):
        return "✊ Fist"

    return "❓ Unknown"


# -----------------------------------
# CAMERA
# -----------------------------------

st.subheader("📷 Camera")

camera_image = st.camera_input("Take a picture of your hand")

# -----------------------------------
# PROCESS IMAGE
# -----------------------------------

if camera_image is not None:

    image = Image.open(camera_image).convert("RGB")

    image_array = np.array(image)

    # Process image with MediaPipe

    results = hands.process(image_array)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks

            mp_drawing.draw_landmarks(
                image_array,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Recognize gesture

            gesture = recognize_gesture(
                hand_landmarks.landmark
            )

            st.success(
                f"Detected Gesture: {gesture}"
            )

        # Display processed image

        st.image(
            image_array,
            caption="Detected Hand",
            use_container_width=True
        )

    else:

        st.warning(
            "No hand detected. Please try again with your hand clearly visible."
        )

# -----------------------------------
# TECHNOLOGY
# -----------------------------------

st.divider()

st.subheader("🛠️ Technology Used")

st.write(
    "Python, Streamlit, MediaPipe, NumPy and Computer Vision."
)

st.write(
    "MediaPipe detects hand landmarks and the landmark positions "
    "are analyzed to recognize the gesture."
)
