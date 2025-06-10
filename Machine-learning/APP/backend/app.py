from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = joblib.load('pneumonia_ensemble_model.pkl')

# Preprocess image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")  # grayscale
    img = img.resize((13, 13))  # size that gives around 169 features
    img_array = np.array(img) / 255.0
    img_array = img_array.flatten().reshape(1, -1)  # shape (1, 169)
    return img_array[:, :158]  # match model’s input (158 features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['xray']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = preprocess_image(filepath)
            prediction = model.predict(image)[0]
            result = "He/She has Pneumonia" if prediction == 0 else "No Pneumonia detected"
            return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

    return jsonify({"error": "Invalid request"})

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
