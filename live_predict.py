import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
# from tensorflow.keras.models import load_model  # Uncomment if using real model

# Simulated class labels
class_labels = ['Normal', 'Vitamin A Deficiency', 'Vitamin B12 Defqiciency', 'Vitamin D Deficiency']

# Load pretrained model if available (optional)
# model = load_model('model.h5')  # Uncomment if using actual trained model

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    # If using real model:
    # processed = preprocess_image(image)
    # prediction = model.predict(processed)
    # predicted_class = np.argmax(prediction)
    # confidence = prediction[0][predicted_class]

    # Simulated output
    predicted_class = random.randint(0, len(class_labels)-1)
    confidence = round(random.uniform(0.75, 0.99), 2)

    return class_labels[predicted_class], confidence

# Open webcam
cap = cv2.VideoCapture(0)
print("📸 Press 'c' to capture an image and check for vitamin deficiency.")
print("❌ Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access the camera.")
        break

    cv2.imshow("Live Feed - Press 'c' to Capture", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        # Capture and process
        captured_image = frame.copy()
        result, confidence = predict(captured_image)

        # Convert to RGB for display
        img_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f'Prediction: {result}\nConfidence: {confidence}')
        plt.axis('off')
        plt.show()

        print(f"\nPredicted: {result} (Confidence: {confidence})")

    elif key & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()



