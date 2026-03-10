# collect.py
import sys
import time
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

def extract_landmarks_from_result(result):
    """Retourne un vecteur (63,) à partir du HandLandmarkerResult, ou None."""
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]  # une main
    coords = []
    for p in lm:
        coords.extend([p.x, p.y, p.z])
    return np.array(coords, dtype=np.float32)

def main():
    if len(sys.argv) < 3:
        print("Usage: python collect.py <nom_geste> <nb_samples>")
        sys.exit(1)

    gesture_name = sys.argv[1]
    num_samples = int(sys.argv[2])

    data_dir = Path("data_test") / gesture_name
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / "samples.npy"

    print(f"Geste : {gesture_name}")
    print(f"Nombre d'échantillons à collecter : {num_samples}")

    mp_result = {"value": None}

    def callback(result, image, timestamp_ms):
        mp_result["value"] = result

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=callback,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: webcam non disponible.")
        sys.exit(1)

    print("Appuie sur 's' pour commencer la capture, 'q' pour quitter.")
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        collected = []
        started = False
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            # Affichage info
            text = f"{gesture_name} - {len(collected)}/{num_samples}"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Collecte", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('s'):
                started = True
                print("Début de la capture...")

            if started and mp_result["value"] is not None:
                vec = extract_landmarks_from_result(mp_result["value"])
                if vec is not None:
                    collected.append(vec)
                    print(f"Sample {len(collected)}/{num_samples}")
                    time.sleep(0.1)  # petite pause pour varier les frames

            if len(collected) >= num_samples:
                print("Collecte terminée.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if collected:
        collected = np.stack(collected)
        if save_path.exists():
            old = np.load(save_path)
            collected = np.concatenate([old, collected], axis=0)
        np.save(save_path, collected)
        print(f"✅ Sauvegardé : {save_path} ({collected.shape[0]} samples)")
    else:
        print("Aucun sample collecté.")

if __name__ == "__main__":
    main()
