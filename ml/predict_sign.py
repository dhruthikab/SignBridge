
import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import subprocess
import queue

from pathlib import Path
from collections import deque


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).parent

HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = BASE_DIR / "models" / "sequence_classifier.joblib"

FRAMES_PER_SAMPLE = 30
FEATURES_PER_FRAME = 63

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(f"Hand model not found: {HAND_MODEL_PATH}")

if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")


# ============================================================
# LOAD CLASSIFIER
# ============================================================

classifier = joblib.load(CLASSIFIER_PATH)



def speak_in_background(text):
    safe_text = text.replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speaker.Speak('{safe_text}'); "
        "$speaker.Dispose()"
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", command],
        creationflags=subprocess.CREATE_NO_WINDOW
    )



print("Loaded sign classifier.")
print("Labels:", classifier.classes_)
print("Press SPACE to capture a sequence.")
print("Press Q to quit.")


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

sequence = []
recording = False
prediction_text = "Press SPACE to record"
confidence_text = ""


# ============================================================
# PREDICTION LOOP
# ============================================================

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            hand_detected = len(result.hand_landmarks) > 0

            if hand_detected:
                landmarks = result.hand_landmarks[0]
                height, width, _ = frame.shape

                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame, (x, y), 4, (0, 255, 0), -1
                    )

                # Collect one sequence when recording
                if recording:
                    frame_values = []

                    for landmark in landmarks:
                        frame_values.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    sequence.append(frame_values)

                    if len(sequence) >= FRAMES_PER_SAMPLE:
                        features = np.asarray(
                            sequence[:FRAMES_PER_SAMPLE],
                            dtype=np.float32
                        ).reshape(1, -1)

                        probabilities = classifier.predict_proba(features)[0]
                        best_index = int(np.argmax(probabilities))

                        
                        prediction_text = str(classifier.classes_[best_index])

                        # Convert the model label into user-friendly text
                        spoken_text = {
                             "HELLO": "Hello!",
                             "BYE-BYE": "Bye bye!"
                        }.get(prediction_text, prediction_text.replace("_", " "))

                        # Display friendly text
                        prediction_text = spoken_text

                        # Speak the recognized sign
                        print(f"Speaking: {spoken_text}")
                        speak_in_background(spoken_text)

                        confidence_text = (
                            f"Model score: "
                            f"{probabilities[best_index] * 100:.1f}%"
                        )

                        print(
                            f"Prediction: {prediction_text} | "
                            f"{confidence_text}"
                        )

                        sequence = []
                        recording = False

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                sequence = []
                recording = True
                prediction_text = "Recording sequence..."
                confidence_text = ""
                print("Recording 30 detected-hand frames...")

            elif key == ord("q"):
                break

            cv2.putText(
                frame,
                f"Prediction: {prediction_text}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                confidence_text,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            if recording:
                cv2.putText(
                    frame,
                    f"Frames: {len(sequence)}/{FRAMES_PER_SAMPLE}",
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frame,
                "SPACE: record | Q: quit",
                (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("SignBridge - Sign Prediction", frame)

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Prediction window closed.")
