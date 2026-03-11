# Hand-Gesture Hospital Assistant

## Project

This project is designed for **hospital rooms** to help patients with reduced mobility or speech difficulties communicate using **simple hand gestures**.  
A webcam points at the patient’s hand; the system uses **MediaPipe** to detect 21 hand landmarks and a **CNN 1D model** to classify each gesture (e.g. *call nurse*, *yes*, *no*, *pain*) in **real time** and display the corresponding message.

---

## Pre‑requisites

- Python 3.9+ installed
- A webcam connected to the machine
- Git (optional, to clone the repo)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

Windows (PowerShell):
```bash
python -m venv venv
.\venv\Scripts\activate
```

Linux / macOS:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

    The file hand_landmarker.task (MediaPipe hand model) must be present in the project root.

---

## Launching

The project is organised around three main scripts:
- collect.py – collect gesture samples with the webcam
- train.py (and/or train.py) – train the gesture recognition model
- main.py – run real‑time gesture recognition

### 1. Collect training data

For each gesture you want to support (e.g. call_nurse, yes, no, pain):
```bash
python collect.py call_nurse 50 # Arguments: gesture_name number_of_samples
python collect.py yes 50
python collect.py no 50
python collect.py pain 50
```

Each command:
- opens the webcam
- records the specified number of samples
- saves landmark vectors into data/<gesture>/samples.npy

You should end up with:
```text
data/
 ├── call_nurse/samples.npy
 ├── yes/samples.npy
 ├── no/samples.npy
 └── pain/samples.npy
```

    (Optional) Collect separate test data with option `--output data_test`

### 2. Train the model

CNN 1D:
```bash
python train.py
```

This:
- loads data from data/
- encodes labels and reshapes features to (21, 3)
- splits into train / validation
- trains a 1D‑CNN with early stopping
- evaluates on data_test/ if available

saves:
- models/gesture_cnn_tf.h5 (trained model)
- models/label_encoder.pkl (label encoder)

---

## Usage

Run real‑time gesture recognition
```bash
python main.py
```

What happens:
1. The script loads the CNN model and label encoder from models/.
2. A webcam window opens.
3. MediaPipe detects the hand and extracts 21 landmarks.
4. Landmarks are converted into a feature vector and passed to the CNN 1D.
5. If the highest predicted probability is above the confidence threshold (default 0.7), the corresponding gesture name is displayed on the video feed.

Controls & behaviour
- Press q to quit the application.
- If no hand is detected, no gesture is displayed.
- If the prediction confidence is below 0.7, the system stays in a “no gesture” state to avoid accidental activations.

---

## Typical workflow

1. Define the gestures and labels you want (e.g. call_nurse, yes, no, pain).
2. Use collect.py to record a balanced dataset for each gesture.
3. Run train_tf.py to train and evaluate the CNN model.
4. Launch main.py to test the system in real time with the webcam.
5. Integrate the predicted gestures with a hospital UI (alerts, lights, bed controls, etc.) as needed.
