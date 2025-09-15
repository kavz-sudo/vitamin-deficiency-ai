import os
import sys
import cv2
import numpy as np
import skfuzzy as fuzz
from tensorflow.keras.models import load_model

# ---------- Paths that work in .py AND in PyInstaller EXE ----------
def base_dir():
    if getattr(sys, 'frozen', False):
        # Running from bundled EXE
        return sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    # Running as normal .py
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = base_dir()
MODEL_FILENAME = "vitamin_model_mobilenet.h5"   # <- use the exact filename you want
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), MODEL_FILENAME)
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------- Load model (with clear error if missing) ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}\n"
        f"Place '{MODEL_FILENAME}' next to your app.exe"
    )

model = load_model(MODEL_PATH)

# ---------- Labels (must match training order) ----------
class_labels = ['Normal', 'Vitamin A Deficiency', 'Vitamin B12 Deficiency', 'Vitamin D Deficiency']

# ---------- Food recommendations ----------
food_recommendations = {
    "Vitamin A Deficiency": [
        "Carrots", "Sweet potatoes", "Spinach", "Kale", "Mango", "Pumpkin", "Eggs", "Milk"
    ],
    "Vitamin B12 Deficiency": [
        "Fish (salmon, tuna)", "Eggs", "Milk", "Cheese", "Yogurt", "Fortified cereals", "Chicken", "Beef"
    ],
    "Vitamin D Deficiency": [
        "Fatty fish (salmon, mackerel)", "Egg yolks", "Fortified milk", "Fortified cereals", "Mushrooms", "Cheese"
    ]
}

# ---------- Preprocess ----------
def preprocess_image(image):
    if image is None:
        raise ValueError("Empty image passed to preprocess_image.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MobileNet usually trained on RGB
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# ---------- Predict ----------
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    blob = preprocess_image(img)
    # IMPORTANT: verbose=0 avoids Keras writing to missing console in --noconsole EXE
    probs = model.predict(blob, verbose=0)
    class_id = int(np.argmax(probs[0]))
    confidence = float(probs[0][class_id])
    return class_labels[class_id], confidence

# ---------- Robust webcam open (tries 0 then 1) ----------
def open_camera():
    indexes = [0, 1]
    for idx in indexes:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return None

def capture_image(part_name, save_path):
    cap = open_camera()
    if cap is None:
        raise RuntimeError("No camera available. Plug a webcam and try again.")

    print(f"\n📸 Position your {part_name.upper()} in front of the camera. Press 's' to capture, 'q' to cancel.")
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Camera read failed.")
        cv2.imshow(f"Capture {part_name}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(save_path, frame)
            print(f"✅ Saved {part_name} image at {save_path}")
            break
        elif key == ord('q'):
            print("❌ Capture cancelled by user.")
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------- Fuzzy decision ----------
def fuzzy_decision(confidences):
    # confidences: list of floats in [0,1]
    x = np.linspace(0, 1, 101)
    high = fuzz.trimf(x, [0.6, 0.8, 1.0])
    medium = fuzz.trimf(x, [0.3, 0.5, 0.7])
    low = fuzz.trimf(x, [0.0, 0.2, 0.4])

    total_score = 0.0
    for conf in confidences:
        conf = float(max(0.0, min(1.0, conf)))
        hv = fuzz.interp_membership(x, high, conf)
        mv = fuzz.interp_membership(x, medium, conf)
        lv = fuzz.interp_membership(x, low, conf)
        total_score += (2 * hv + 1 * mv + 0 * lv)

    score_normalized = total_score / (2 * max(1, len(confidences)))

    if score_normalized > 0.75:
        return "🔴 High Deficiency Risk"
    elif score_normalized > 0.4:
        return "🟠 Moderate Risk"
    else:
        return "🟢 Low Risk"

# ---------- Main ----------
def run_full_diagnosis():
    body_parts = ["eye", "lip", "tongue", "nail"]
    predictions = []
    confidences = []

    print("\n🩺 VITAMIN DEFICIENCY DETECTION SYSTEM")
    print("--------------------------------------")

    for part in body_parts:
        img_path = os.path.join(IMAGES_DIR, f"{part}.jpg")
        capture_image(part, img_path)              # 1) capture
        if not os.path.exists(img_path):
            print(f"⚠️ Skipped {part} (no image saved).")
            continue
        label, conf = predict_image(img_path)      # 2) predict
        predictions.append((part, label, conf))
        confidences.append(conf)

    print("\n🔍 ANALYSIS RESULTS:")
    final_labels = set()
    for part, label, conf in predictions:
        print(f"{part.capitalize():<8}: {label} ({conf*100:.2f}%)")
        if label != "Normal":
            final_labels.add(label)

    final_diagnosis = fuzzy_decision(confidences)
    print(f"\n🧠 FINAL DIAGNOSIS: {final_diagnosis}")

    if not final_labels:
        print("✅ Overall Status: Normal")
    else:
        deficiencies = ", ".join(sorted(final_labels))
        print(f"⚠️ Overall Status: Deficiency Detected → {deficiencies}")
        print("\n🍎 Recommended Foods:")
        for deficiency in sorted(final_labels):
            foods = ", ".join(food_recommendations.get(deficiency, []))
            if foods:
                print(f"   {deficiency}: {foods}")

if __name__ == "__main__":
    try:
        run_full_diagnosis()
    except Exception as e:
        # When running with --noconsole, show a GUI error so users see it.
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk(); root.withdraw()
            messagebox.showerror("Application Error", str(e))
        except Exception:
            pass
        raise
