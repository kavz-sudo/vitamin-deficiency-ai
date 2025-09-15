import cv2
import numpy as np
import os
import skfuzzy as fuzz
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# 🔹 Load trained model
model = load_model("vitamin_deficiency_model_mobilenet.h5")

# 🔹 Class labels
class_labels = ['Normal', 'Vitamin A Deficiency', 'Vitamin B12 Deficiency', 'Vitamin D Deficiency']

# 🔹 Food recommendations
food_recommendations = {
    "Vitamin A Deficiency": ["Carrots", "Sweet potatoes", "Spinach", "Kale", "Mango", "Pumpkin", "Red bell peppers", "Eggs", "Milk"],
    "Vitamin B12 Deficiency": ["Fish (salmon, tuna)", "Eggs", "Milk", "Cheese", "Yogurt", "Fortified cereals", "Chicken", "Beef"],
    "Vitamin D Deficiency": ["Fatty fish (salmon, mackerel)", "Egg yolks", "Fortified milk", "Fortified cereals", "Mushrooms", "Cheese"]
}

# 🔹 Translations (English, Hindi, Tamil)
translations = {
    "en": {
        "report_title": "Vitamin Deficiency Detection Report",
        "patient_name": "Patient Name",
        "analysis_results": "Analysis Results",
        "final_diagnosis": "Final Diagnosis",
        "recommended_foods": "Recommended Foods",
        "status_normal": "✅ Overall Status: Normal",
        "status_deficiency": "⚠️ Overall Status: Deficiency Detected → ",
    },
    "hi": {
        "report_title": "विटामिन की कमी की जांच रिपोर्ट",
        "patient_name": "रोगी का नाम",
        "analysis_results": "विश्लेषण परिणाम",
        "final_diagnosis": "अंतिम निदान",
        "recommended_foods": "अनुशंसित भोजन",
        "status_normal": "✅ कुल स्थिति: सामान्य",
        "status_deficiency": "⚠️ कुल स्थिति: कमी पाई गई → ",
    },
    "ta": {
        "report_title": "விட்டமின் குறைபாடு கண்டறிதல் அறிக்கை",
        "patient_name": "நோயாளியின் பெயர்",
        "analysis_results": "பகுப்பாய்வு முடிவுகள்",
        "final_diagnosis": "இறுதி நோயறிதல்",
        "recommended_foods": "பரிந்துரைக்கப்பட்ட உணவுகள்",
        "status_normal": "✅ மொத்த நிலை: சாதாரணம்",
        "status_deficiency": "⚠️ மொத்த நிலை: குறைபாடு கண்டறியப்பட்டது → ",
    }
}

# 🔸 Preprocessing
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# 🔸 Predict label
def predict_image(image_path):
    img = cv2.imread(image_path)
    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    return class_labels[class_id], confidence

# 🔸 Capture Image
def capture_image(part_name, save_path):
    cap = cv2.VideoCapture(1)  # Change index if multiple cameras
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

# 🔸 Fuzzy decision
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
        return "🔴 High Deficiency Risk – Consult Doctor"
    elif score_normalized > 0.4:
        return "🟠 Moderate Risk – Consult Doctor"
    else:
        return "🟢 Low Risk – Maintain Health"

# 🔸 Save PDF report
def save_report_pdf(patient_folder, patient_name, predictions, final_diagnosis, final_labels, lang="en"):
    t = translations[lang]
    report_path = os.path.join(patient_folder, f"{patient_name}_report.pdf")
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, t["report_title"])

    # Patient info
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"{t['patient_name']}: {patient_name}")

    # Predictions
    c.drawString(50, height - 140, t["analysis_results"] + ":")
    y = height - 160
    for part, label, conf in predictions:
        c.drawString(70, y, f"{part.capitalize()}: {label} ({conf*100:.2f}%)")
        y -= 20

    # Final Diagnosis
    c.drawString(50, y - 20, f"{t['final_diagnosis']}: {final_diagnosis}")
    y -= 60

    # Status
    if not final_labels:
        c.drawString(50, y, t["status_normal"])
    else:
        deficiencies = ", ".join(final_labels)
        c.drawString(50, y, t["status_deficiency"] + deficiencies)
        y -= 40

        # Food recommendations
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, t["recommended_foods"] + ":")
        y -= 20
        c.setFont("Helvetica", 12)
        for deficiency in final_labels:
            if deficiency in food_recommendations:
                foods = ", ".join(food_recommendations[deficiency])
                c.drawString(70, y, f"{deficiency}: {foods}")
                y -= 20

    c.save()
    print(f"📄 PDF Report saved at {report_path}")

# 🔹 Main diagnosis
def run_full_diagnosis():
    patient_name = input("Enter patient name: ")
    lang = input("Choose language (en=English, hi=Hindi, ta=Tamil): ").strip().lower()
    if lang not in translations:
        lang = "en"

    patient_folder = os.path.join("images", patient_name)
    os.makedirs(patient_folder, exist_ok=True)

    body_parts = ["eye", "skin", "tongue", "nail"]
    predictions = []
    confidences = []

    print("🩺 VITAMIN DEFICIENCY DETECTION SYSTEM")
    print("--------------------------------------")

    for part in body_parts:
        path = os.path.join(patient_folder, f"{part}.jpg")
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

    save_report_pdf(patient_folder, patient_name, predictions, final_diagnosis, final_labels, lang)

# Run
if __name__ == "__main__":
    run_full_diagnosis()
