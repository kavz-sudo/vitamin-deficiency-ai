import cv2
import numpy as np
import skfuzzy as fuzz
from tensorflow.keras.models import load_model

# 🔹 Load trained model (you must train & provide this file)
model = load_model("vitamin_deficiency_model_mobilenet.h5")

# 🔹 Your class labels
class_labels = ['Normal', 'Vitamin A Deficiency', 'Vitamin B12 Deficiency', 'Vitamin D Deficiency']

# 🔸 Preprocessing for MobileNet
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# 🔸 Make prediction and return label + confidence
def predict_image(image_path):
    img = cv2.imread(image_path)
    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    return class_labels[class_id], confidence

# 🔸 Live webcam capture for each organ
def capture_image(part_name, save_path):
    cap = cv2.VideoCapture(1)
    print(f"\n📸 Place your {part_name.upper()} in front of the camera. Press 's' to capture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break
        cv2.imshow(f"Capture {part_name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(save_path, frame)
            break
    cap.release()
    cv2.destroyAllWindows()

# 🔸 Fuzzy logic decision (simplified rule-based)
def fuzzy_decision(confidences):
    x = np.arange(0, 1.1, 0.1)
    high = fuzz.trimf(x, [0.6, 0.8, 1.0])
    medium = fuzz.trimf(x, [0.3, 0.5, 0.7])
    low = fuzz.trimf(x, [0.0, 0.2, 0.4])

    total_score = 0
    for conf in confidences:
        high_val = fuzz.interp_membership(x, high, conf)
        med_val = fuzz.interp_membership(x, medium, conf)
        low_val = fuzz.interp_membership(x, low, conf)
        total_score += (2 * high_val + 1 * med_val + 0 * low_val)

    score_normalized = total_score / (2 * len(confidences))

    if score_normalized > 0.75:
        return "🔴 High Deficiency Risk consult doctor "
    elif score_normalized > 0.4:
        return "🟠 Moderate Risk consult doctor "
    else:
        return "🟢 Low Risk maintain your health"

# 🔹 Main program
def run_full_diagnosis():
    body_parts = ["eye", "skin", "tongue", "nail"]
    predictions = []
    confidences = []

    print("🩺 VITAMIN DEFICIENCY DETECTION SYSTEM")
    print("--------------------------------------")

    for part in body_parts:
        path = f"images/{part}.jpg"
        capture_image(part, path)
        label, confidence = predict_image(path)
        predictions.append((part, label, confidence))
        confidences.append(confidence)

    print("\n🔍 ANALYSIS RESULTS:")
    for part, label, conf in predictions:
        print(f"{part.capitalize():<8}: {label} ({conf*100:.2f}%)")

    final_diagnosis = fuzzy_decision(confidences)
    print(f"\n🧠 FINAL DIAGNOSIS: {final_diagnosis}")

# 🔸 Run it
if __name__ == "__main__":
    run_full_diagnosis()
