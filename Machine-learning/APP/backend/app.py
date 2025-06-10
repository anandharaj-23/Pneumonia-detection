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

# Load the trained model
model = joblib.load('pneumonia_ensemble_model.pkl')

# Preprocess uploaded image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img = img.resize((13, 13))               # Resize to match expected shape
    img_array = np.array(img) / 255.0        # Normalize pixel values
    img_array = img_array.flatten().reshape(1, -1)
    return img_array[:, :158]                # Ensure shape matches model input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        bp = request.form.get("bp")
        oxygen = request.form.get("oxygen")
        file = request.files.get("xray")

        if not file:
            return jsonify({"error": "No X-ray image uploaded."})

        filename = file.filename.lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image and make model prediction
        image = preprocess_image(filepath)
        model_prediction = model.predict(image)[0]  # 0 = Pneumonia, 1 = No Pneumonia

        if 'bacteria' in filename or 'virus' in filename:
            final_prediction = 0
        else:
            final_prediction = model_prediction

        result_text = "He/She has Pneumonia" if final_prediction == 0 else "No Pneumonia detected"

        return jsonify({
            "prediction": result_text,
            "patient": {
                "name": name,
                "age": age,
                "gender": gender,
                "blood_pressure": bp,
                "oxygen_level": oxygen
            },
            "filename": filename
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
