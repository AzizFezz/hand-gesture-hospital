import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

#  CONFIG 
MODEL_PATH = "hand_landmarker.task"           # modèle MediaPipe
CONFIDENCE_THRESHOLD = 0.7                    # seuil pour accepter un geste

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

latest_result = None

#  MEDIAPIPE CALLBACK 

def result_callback(result, image, timestamp_ms):
    global latest_result
    latest_result = result

#  DESSIN DU SQUELETTE 

def to_pixel(x_norm, y_norm, w, h):
    x = min(max(x_norm, 0.0), 1.0)
    y = min(max(y_norm, 0.0), 1.0)
    return int(x * w), int(y * h)

def draw_hand_landmarks_manual(image_bgr, hand_landmarks_list):
    h, w = image_bgr.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        pts = [to_pixel(lm.x, lm.y, w, h) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(image_bgr, pts[a], pts[b], (0, 255, 0), 2)
        for (x, y) in pts:
            cv2.circle(image_bgr, (x, y), 4, (0, 0, 255), -1)

#  EXTRACTION FEATURES 63D 

def extract_vector_from_latest():
    """
    Retourne un vecteur (1, 63) à partir du dernier résultat MediaPipe,
    ou None s'il n'y a pas de main détectée.
    """
    if latest_result is None or not latest_result.hand_landmarks:
        return None

    lm = latest_result.hand_landmarks[0]  # une seule main
    coords = []
    for p in lm:
        coords.extend([p.x, p.y, p.z])

    vec = np.array(coords, dtype=np.float32).reshape(1, -1)  # (1, 63)
    return vec

#  CHARGEMENT CNN + ENCODER 

def load_cnn_model():
    models_dir = Path("models")
    model_path = models_dir / "gesture_cnn_tf.h5"
    le_path = models_dir / "label_encoder.pkl"

    if not model_path.exists() or not le_path.exists():
        print("⚠️  Modèle TF ou LabelEncoder introuvable.")
        print("    Lance d'abord : python train_tf.py")
        return None, None

    model = tf.keras.models.load_model(model_path)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    return model, le

#  MAIN LOOP 

def main():
    # Charger le modèle CNN et le label encoder
    model, le = load_cnn_model()
    if model is None:
        return

    # Configurer MediaPipe HandLandmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Hand Tracking + CNN TF Gesture Recognition. Press 'q' to exit.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            # Miroir + conversion
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Envoi à MediaPipe (asynchrone)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            gesture_name = ""

            # Extraire vecteur 63D si une main est détectée
            vec = extract_vector_from_latest()

            if vec is not None:
                # Adapter au CNN : (1,63) -> (1,21,3)
                x_cnn = vec.reshape(1, 21, 3)
                preds = model.predict(x_cnn, verbose=0)          # (1, num_classes)
                prob = float(np.max(preds))
                pred_idx = int(np.argmax(preds, axis=1)[0])
                predicted_label = le.inverse_transform([pred_idx])[0]

                # Gestion de l'état vide via un seuil de confiance
                if prob >= CONFIDENCE_THRESHOLD:
                    gesture_name = predicted_label
                else:
                    gesture_name = ""  # geste pas clair → état vide

            # Dessiner squelette si présent
            if latest_result and latest_result.hand_landmarks:
                draw_hand_landmarks_manual(frame, latest_result.hand_landmarks)

            # Afficher le nom du geste uniquement si non vide
            if gesture_name:
                cv2.putText(
                    frame,
                    f"{gesture_name}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("Hand Tracking - CNN TF", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
