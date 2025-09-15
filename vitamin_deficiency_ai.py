import cv2
import numpy as np
import skfuzzy as fuzz
from tensorflow.keras.models import load_model

# 🔹 Load trained model
model = load_model("vitamin_deficiency_model_mobilenet.h5")

# 🔹 Class labels
class_labels = ['Normal', 'Vitamin A Deficiency', 'Vitamin B12 Deficiency', 'Vitamin D Deficiency']

# 🔹 Food recommendations dictionary
food_recommendations = {
    "Vitamin A Deficiency": [
        "Carrots", "Sweet potatoes", "Spinach", "Kale", "Mango", "Pumpkin", "Red bell peppers", "Eggs", "Milk"
    ],
    "Vitamin B12 Deficiency": [
        "Fish (salmon, tuna)", "Eggs", "Milk", "Cheese", "Yogurt", "Fortified cereals", "Chicken", "Beef"
    ],
    "Vitamin D Deficiency": [
        "Fatty fish (salmon, mackerel)", "Egg yolks", "Fortified milk", "Fortified cereals", "Mushrooms", "Cheese"
    ]
}

# 🔸 Preprocessing for MobileNet
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# 🔸 Predict label & confidence
def predict_image(image_path):
    img = cv2.imread(image_path)
    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    return class_labels[class_id], confidence

# 🔸 Live webcam capture
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

# 🔸 Fuzzy logic decision
def fuzzy_decision(confidences):
    x = np.arange(0, 1.1, 0.1)
    high = fuzz.trimf(x, [0.6, 0.8, 1.0])
    medium = fuzz.trimf(x, [0.3, 0.5, 0.7])
    low = fuzz.trimf(x, [0.0, 0.2, 0.4])

    total_score = 0
    for conf in confidences:
        high_val = fuzz.interp_membership(x, high,
         conf)
        med_val = fuzz.interp_membership(x, medium, conf)
        low_val = fuzz.interp_membership(x, low, conf)
        total_score += (2 * high_val + 1 * med_val + 0 * low_val)

    score_normalized = total_score / (2 * len(confidences))

    if score_normalized > 0.75:
        return "🔴 High Deficiency Risk consult docter"
    elif score_normalized > 0.4:
        return "🟠 Moderate Risk  consult docter"
    else:
        return "🟢 Low Risk maintain you health"

# 🔹 Main detection
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
    final_labels = set()

    for part, label, conf in predictions:
        print(f"{part.capitalize():<8}: {label} ({conf*100:.2f}%)")
        if label != "Normal":
            final_labels.add(label)

    final_diagnosis = fuzzy_decision(confidences)
    print(f"\n🧠 FINAL DIAGNOSIS: {final_diagnosis}")

    # Consolidated overall result with food recommendations
    if not final_labels:
        print("✅ Overall Status: Normal")
    else:
        deficiencies = ", ".join(final_labels)
        print(f"⚠️ Overall Status: Deficiency Detected → {deficiencies}")
        print("\n🍎 Recommended Foods:")
        for deficiency in final_labels:
            if deficiency in food_recommendations:
                foods = ", ".join(food_recommendations[deficiency])
                print(f"   {deficiency}: {foods}")

# 🔸 Run program
if __name__ == "__main__":
    run_full_diagnosis()
