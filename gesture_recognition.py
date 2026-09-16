import cv2
import mediapipe as mp
import math
# Calculate distance between two landmarks
def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )
# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Hand Gesture Recognition Started")
print("Press Q to quit.")
# Main loop
while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    # Flip image for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect hand
    results = hands.process(rgb_frame)

    gesture = "No Hand Detected"

    # If hand detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark

            # Hand size reference
            hand_size = distance(
                landmarks[0],
                landmarks[9]
            )

            # Check four fingers
            index_up = (
                landmarks[8].y <
                landmarks[6].y
            )

            middle_up = (
                landmarks[12].y <
                landmarks[10].y
            )

            ring_up = (
                landmarks[16].y <
                landmarks[14].y
            )

            pinky_up = (
                landmarks[20].y <
                landmarks[18].y
            )

            # Check thumb extension
            thumb_extended = (
                distance(
                    landmarks[4],
                    landmarks[5]
                ) > hand_size * 0.35
            )

            # Check thumb pointing upward
            thumb_pointing_up = (
                landmarks[4].y <
                landmarks[3].y and
                landmarks[4].y <
                landmarks[2].y
            )

            # -----------------------------
            # Gesture classification
            # -----------------------------

            # Open Hand
            if (
                index_up and
                middle_up and
                ring_up and
                pinky_up and
                thumb_extended
            ):
                gesture = "Open Hand"

            # Victory / Peace
            elif (
                index_up and
                middle_up and
                not ring_up and
                not pinky_up
            ):
                gesture = "Victory / Peace"

            # Thumbs Up
            elif (
                not index_up and
                not middle_up and
                not ring_up and
                not pinky_up and
                thumb_pointing_up
            ):
                gesture = "Thumbs Up"

            # Fist
            elif (
                not index_up and
                not middle_up and
                not ring_up and
                not pinky_up
            ):
                gesture = "Fist"

            else:
                gesture = "Unknown Gesture"

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # -----------------------------
    # Display gesture
    # -----------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (500, 80),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show webcam
    cv2.imshow(
        "AI Hand Gesture Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

print("Program ended.")