from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ==========================
# Load Files
# ==========================

model = joblib.load("brain_tumor_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")


@app.route("/")
def home():
    return {
        "message": "OncoVision AI Backend Running Successfully"
    }


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        # Encode categorical values

        gender = encoders["Gender"].transform([data["Gender"]])[0]
        country = encoders["Country"].transform([data["Country"]])[0]
        genetic = encoders["Genetic_Risk"].transform([data["Genetic_Risk"]])[0]
        smoking = encoders["Smoking_History"].transform([data["Smoking_History"]])[0]
        alcohol = encoders["Alcohol_Consumption"].transform([data["Alcohol_Consumption"]])[0]
        radiation = encoders["Radiation_Exposure"].transform([data["Radiation_Exposure"]])[0]
        headinjury = encoders["Head_Injury_History"].transform([data["Head_Injury_History"]])[0]
        chronic = encoders["Chronic_Illness"].transform([data["Chronic_Illness"]])[0]
        bp = encoders["Blood_Pressure"].transform([data["Blood_Pressure"]])[0]
        diabetes = encoders["Diabetes"].transform([data["Diabetes"]])[0]
        family = encoders["Family_History"].transform([data["Family_History"]])[0]
        severity = encoders["Symptom_Severity"].transform([data["Symptom_Severity"]])[0]

        features = np.array([[
            data["Age"],
            gender,
            country,
            genetic,
            smoking,
            alcohol,
            radiation,
            headinjury,
            chronic,
            bp,
            diabetes,
            family,
            severity
        ]])

        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(features))) * 100

        return jsonify({

            "prediction": int(prediction),

            "confidence": round(confidence,2) if confidence else None

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)