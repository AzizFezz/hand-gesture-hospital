# train_tf.py
"""
Entraînement CNN 1D : data/ → train + validation
data_test/ → test final séparé (évaluation après entraînement)
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
import pickle


# --------------------------------------------------
# 1. Chargement dataset depuis data/ (train+val)
# --------------------------------------------------
def load_train_val_dataset(data_root="data"):
    data_root = Path(data_root)
    X_list, y_list = [], []

    if not data_root.exists():
        print("❌ Pas de dossier 'data/'. Lance d'abord collect.py.")
        return None, None, None

    print("📂 Chargement TRAIN+VAL depuis", data_root)

    for gesture_dir in sorted(data_root.iterdir()):
        if not gesture_dir.is_dir():
            continue

        samples_path = gesture_dir / "samples.npy"
        if not samples_path.exists():
            continue

        samples = np.load(samples_path)  # (N_samples, 63)
        n = samples.shape[0]
        if n == 0:
            continue

        print(f"  - {gesture_dir.name}: {n} samples")
        X_list.append(samples)
        y_list.extend([gesture_dir.name] * n)

    if not X_list:
        print("❌ Aucune donnée dans 'data/'.")
        return None, None, None

    X = np.concatenate(X_list, axis=0)           # (N, 63)
    y = np.array(y_list)                         # (N,)
    label_names = sorted(list(set(y_list)))

    print(f"\n✅ Dataset train+val : {X.shape[0]} samples")
    return X, y, label_names


# --------------------------------------------------
# 2. Chargement dataset TEST séparé depuis data_test/
# --------------------------------------------------
def load_test_dataset(data_root="data_test", label_encoder=None):
    data_root = Path(data_root)
    if not data_root.exists():
        print("⚠️  Dossier 'data_test/' non trouvé.")
        return None, None

    X_list, y_list = [], []
    print("\n📂 Chargement TEST depuis", data_root)

    for gesture_dir in sorted(data_root.iterdir()):
        if not gesture_dir.is_dir():
            continue

        samples_path = gesture_dir / "samples.npy"
        if not samples_path.exists():
            continue

        samples = np.load(samples_path)  # (N_samples, 63)
        n = samples.shape[0]
        print(f"  - {gesture_dir.name}: {n} samples")

        X_list.append(samples)
        y_list.extend([gesture_dir.name] * n)

    if not X_list:
        print("⚠️  Aucune donnée dans 'data_test/'.")
        return None, None

    X = np.concatenate(X_list, axis=0)
    y_str = np.array(y_list)
    
    # Encoder avec le LabelEncoder déjà entraîné
    if label_encoder is None:
        print("❌ LabelEncoder manquant pour data_test.")
        return None, None
    
    y = label_encoder.transform(y_str)
    
    return X, y


# --------------------------------------------------
# 3. CNN 1D Model
# --------------------------------------------------
def build_cnn_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(21, 3))

    x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------
# 4. MAIN
# --------------------------------------------------
def main():
    # 1) Charger TRAIN+VAL depuis data/
    X_tv, y_tv_str, label_names = load_train_val_dataset("data")
    if X_tv is None:
        return

    # 2) Encoder les labels
    le = LabelEncoder()
    y_tv = le.fit_transform(y_tv_str)  # (N,)

    num_classes = len(le.classes_)
    print(f"\nNombre de classes : {num_classes}")
    print("Classes :", list(le.classes_))

    # 3) Reshape pour CNN 1D
    X_cnn_tv = X_tv.reshape(-1, 21, 3).astype("float32")

    # 4) Split TRAIN / VALIDATION (80/20 sur data/)
    X_train, X_val, y_train, y_val = train_test_split(
        X_cnn_tv,
        y_tv,
        test_size=0.20,          # 20% de data/ pour validation
        random_state=42,
        stratify=y_tv,
    )

    print("\n📊 Répartition data/ (train + val) :")
    print("  Train :", X_train.shape[0], "samples")
    print("  Val   :", X_val.shape[0],   "samples")

    # 5) Construire le modèle
    model = build_cnn_model(num_classes)
    model.summary()

    # 6) Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5
        ),
    ]

    # 7) Entraînement (train + val)
    print("\n🚀 Début de l'entraînement...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    # 8) Évaluation sur data_test/ (jamais vu par le modèle)
    print("\n🔍 Évaluation sur data_test/ (jeu de test externe)...")
    X_test, y_test = load_test_dataset("data_test", le)
    
    if X_test is not None:
        X_test_cnn = X_test.reshape(-1, 21, 3).astype("float32")
        test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
        
        print(f"\n✅ RÉSULTATS FINAUX :")
        print(f"  Test accuracy (data_test/) : {test_acc:.4f}")
        print(f"  Test loss                 : {test_loss:.4f}")
        
        # Rapport de classification
        y_test_pred = np.argmax(model.predict(X_test_cnn, verbose=0), axis=1)
        print("\nClassification report (data_test/) :")
        print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    else:
        print("⚠️  Pas de data_test/. Résultats sur validation :")
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"  Validation accuracy : {val_acc:.4f}")

    # 9) Sauvegarde
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "gesture_cnn_tf.h5"
    model.save(model_path)
    print(f"\n✅ Modèle sauvé : {model_path}")

    le_path = models_dir / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"✅ LabelEncoder sauvé : {le_path}")


if __name__ == "__main__":
    main()
