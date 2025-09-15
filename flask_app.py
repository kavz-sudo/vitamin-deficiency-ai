import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import cv2
from tensorflow.keras.models import load_model
import skfuzzy as fuzz

# ----- Paths -----
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "vitamin_model_mobilenet.h5"
MODEL_PATH = os.path.join(APP_DIR, MODEL_FILENAME)

# ----- Load model -----
model = load_model(MODEL_PATH)
class_labels = ['Normal', 'Vitamin A Deficiency', 'Vitamin B12 Deficiency', 'Vitamin D Deficiency']

food_recommendations = {
    "Vitamin A Deficiency": ["Carrots","Sweet potatoes","Spinach","Kale","Mango","Pumpkin","Eggs","Milk"],
    "Vitamin B12 Deficiency": ["Fish (salmon, tuna)","Eggs","Milk","Cheese","Yogurt","Fortified cereals","Chicken","Beef"],
    "Vitamin D Deficiency": ["Fatty fish (salmon, mackerel)","Egg yolks","Fortified milk","Fortified cereals","Mushrooms","Cheese"]
}

def preprocess_np(img_np):
    # img_np: RGB numpy array
    img_np = cv2.resize(img_np, (224, 224))
    img_np = img_np.astype("float32") / 255.0
    return np.expand_dims(img_np, axis=0)

def predict_np(img_np):
    blob = preprocess_np(img_np)
    probs = model.predict(blob, verbose=0)
    idx = int(np.argmax(probs[0]))
    conf = float(probs[0][idx])
    return class_labels[idx], conf

def fuzzy_decision(confidences):
    x = np.linspace(0,1,101)
    high = fuzz.trimf(x,[0.6,0.8,1.0])
    med  = fuzz.trimf(x,[0.3,0.5,0.7])
    low  = fuzz.trimf(x,[0.0,0.2,0.4])
    score = 0.0
    for c in confidences:
        hv = fuzz.interp_membership(x, high, c)
        mv = fuzz.interp_membership(x, med, c)
        lv = fuzz.interp_membership(x, low, c)
        score += (2*hv + 1*mv + 0*lv)
    score /= (2*max(1,len(confidences)))
    if score > 0.75: return "🔴 High Deficiency Risk"
    if score > 0.4:  return "🟠 Moderate Risk"
    return "🟢 Low Risk"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Receives a dataURL image from the browser and a "part" string
@app.post("/predict")
def predict():
    data_url = request.form.get("image")
    part = request.form.get("part","unknown")

    if not data_url or not data_url.startswith("data:image"):
        return jsonify({"error": "No image"}), 400

    header, b64data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    label, conf = predict_np(img_np)
    return jsonify({
        "part": part,
        "label": label,
        "confidence": round(conf, 4),
        "foods": food_recommendations.get(label, [])
    })

@app.post("/aggregate")
def aggregate():
    # expects confidences list in form field "confs" as CSV
    confs_csv = request.form.get("confs","")
    if not confs_csv:
        return jsonify({"risk":"🟢 Low Risk"})
    confs = [float(x) for x in confs_csv.split(",") if x]
    return jsonify({"risk": fuzzy_decision(confs)})

@app.route("/video_feed")
def video_feed():
    def gen_frames():
        # 0 = laptop camera, 1 = USB cam
        cam = cv2.VideoCapture(1)  
        while True:
            success, frame = cam.read()
            if not success:
                break
            label, conf, _ = predict_np(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.putText(frame, f"{label} ({conf:.2f})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stop_camera")
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "Camera stopped"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
