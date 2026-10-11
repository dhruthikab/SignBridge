import cv2
import mediapipe as mp
from pathlib import Path


# --------------------------------------------------
# 1. Find the MediaPipe hand landmark model
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# --------------------------------------------------
# 2. Create MediaPipe Hand Landmarker
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# 3. Hand landmark connections
# --------------------------------------------------

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm connections
    (5, 9),
    (9, 13),
    (13, 17),
]


# --------------------------------------------------
# 4. Draw landmarks
# --------------------------------------------------

def draw_hand_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape

    points = []

    # Convert normalized MediaPipe coordinates
    # into pixel coordinates
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        # Draw landmark point
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    # Draw connections between landmarks
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )


# --------------------------------------------------
# 5. Open webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. "
        "Check that your camera is connected and available."
    )


print("Webcam started.")
print("Show your hand to the camera.")
print("Press Q to quit.")


# --------------------------------------------------
# 6. Start Hand Landmarker
# --------------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        # Read frame from webcam
        success, frame = cap.read()

        if not success:
            print("Could not read frame from webcam.")
            break

        # Mirror the camera view
        frame = cv2.flip(frame, 1)

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = landmarker.detect(mp_image)

        # Number of detected hands
        hand_count = len(result.hand_landmarks)

        # Draw detected hands
        for hand_landmarks in result.hand_landmarks:
            draw_hand_landmarks(
                frame,
                hand_landmarks
            )

        # Display number of detected hands
        cv2.putText(
            frame,
            f"Hands detected: {hand_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display instruction
        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show webcam
        cv2.imshow(
            "SignBridge - Hand Landmark Test",
            frame
        )

        # Exit when Q is pressed
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# --------------------------------------------------
# 7. Clean up
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Webcam test finished.")