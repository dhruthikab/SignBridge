import cv2
import csv
import mediapipe as mp
from pathlib import Path



# ============================================================
# SETTINGS
# ============================================================

SIGN_NAME = "PLEASE" \

# Number of complete performances to record
SAMPLES_TO_COLLECT = 30

# Number of consecutive frames in each performance
FRAMES_PER_SAMPLE = 30

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "hand_landmarker.task"
)

# Save sequences to a NEW file.
# Keep the original landmark_data.csv unchanged.
DATA_FILE = (
    Path(__file__).parent.parent
    / "dataset"
    / "sequence_data.csv"
)



# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# Create dataset folder if necessary
DATA_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)



# ============================================================
# CSV HEADER
# ============================================================

header = ["label"]

for frame_index in range(FRAMES_PER_SAMPLE):
    for landmark_index in range(21):
        header.append(
            f"f{frame_index}_x{landmark_index}"
        )
        header.append(
            f"f{frame_index}_y{landmark_index}"
        )
        header.append(
            f"f{frame_index}_z{landmark_index}"
        )


# Create CSV and header if it doesn't exist
if not DATA_FILE.exists():

    with open(
        DATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


print()
print("======================================")
print(" SignBridge - Data Collector")
print("======================================")
print()
print(f"Sign: {SIGN_NAME}")
print(f"Samples: {SAMPLES_TO_COLLECT}")
print()
print("Press SPACE to start collecting.")
print("Press Q to quit.")
print()



# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    samples_collected = 0
    collecting = False
    sequence_frames = []

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

        # Draw detected hand
        if hand_detected:
            hand_landmarks = result.hand_landmarks[0]
            height, width, _ = frame.shape
            points = []

            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

                cv2.circle(
                    frame, (x, y), 5, (0, 255, 0), -1
                )

            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

        key = cv2.waitKey(1) & 0xFF

        # Start one sequence with SPACE
        if key == ord(" ") and not collecting:
            if samples_collected < SAMPLES_TO_COLLECT:
                collecting = True
                sequence_frames = []
                print("\nRecording sign movement...")

        if key == ord("q"):
            break

        # Collect consecutive frames for one performance
        if collecting:

            if hand_detected:
                frame_values = []

                for landmark in result.hand_landmarks[0]:
                    frame_values.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                sequence_frames.append(frame_values)

            else:
                # Missing-hand frames are skipped.
                # Restart this sequence if the hand disappears.
                sequence_frames = []

            cv2.putText(
                frame,
                f"Frames: {len(sequence_frames)}/{FRAMES_PER_SAMPLE}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            if len(sequence_frames) >= FRAMES_PER_SAMPLE:

                row = [SIGN_NAME]

                for frame_values in sequence_frames[:FRAMES_PER_SAMPLE]:
                    row.extend(frame_values)

                with open(
                    DATA_FILE,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(row)

                samples_collected += 1
                collecting = False
                sequence_frames = []

                print(
                    f"Saved performance "
                    f"{samples_collected}/{SAMPLES_TO_COLLECT}"
                )

                if samples_collected >= SAMPLES_TO_COLLECT:
                    print("\nAll sequences collected!")
                    break

        # Display status
        cv2.putText(
            frame,
            f"Sign: {SIGN_NAME}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Performances: {samples_collected}/{SAMPLES_TO_COLLECT}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if not collecting:
            cv2.putText(
                frame,
                "SPACE: record next performance",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "Q: quit",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("SignBridge - Sequence Collector", frame)

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nData collector closed.")
print(f"Dataset saved to: {DATA_FILE}")
