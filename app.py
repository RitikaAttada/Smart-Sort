import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

# Load the trained waste classification model
model = tf.keras.models.load_model("waste_classifier_one.h5")

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define class labels       
class_labels = ["Dry", "Other", "Wet"]

@app.route("/classify", methods=["POST"])
def classify_waste():
    try:
        # Get image from frontend
        file = request.files["file"]
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            return jsonify({"error": "Human face detected! Please use an image without people."})

        # Preprocess the image for waste classification
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[0][predicted_class])

        # Return result
        return jsonify({"category": class_labels[predicted_class], "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
