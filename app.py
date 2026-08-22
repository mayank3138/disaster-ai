from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained AI model
model = joblib.load("disaster_model.pkl")


recommendations = {

    "Flood": {
        "level": "HIGH",
        "message": [
            "Move to higher and safer ground if flooding is occurring.",
            "Avoid walking or driving through floodwater.",
            "Keep drinking water, food, medicines and a flashlight ready.",
            "Follow official evacuation and emergency instructions."
        ]
    },

    "Earthquake": {
        "level": "HIGH",
        "message": [
            "Drop, Cover and Hold On during shaking.",
            "Stay away from windows and objects that may fall.",
            "After shaking stops, check for hazards and follow official instructions.",
            "Be prepared for possible aftershocks."
        ]
    },

    "Cyclone": {
        "level": "HIGH",
        "message": [
            "Stay indoors and away from windows.",
            "Secure loose objects if it is safe to do so.",
            "Keep emergency supplies and charged communication devices ready.",
            "Follow official cyclone warnings and evacuation instructions."
        ]
    },

    "Landslide": {
        "level": "HIGH",
        "message": [
            "Stay away from steep slopes and landslide-prone areas.",
            "If authorities issue an evacuation order, leave promptly.",
            "Watch for changes such as unusual ground movement or blocked drainage.",
            "Do not enter an affected area until authorities say it is safe."
        ]
    },

    "Heatwave": {
        "level": "MEDIUM",
        "message": [
            "Stay hydrated.",
            "Reduce prolonged outdoor activity during the hottest part of the day.",
            "Stay in a cool or well-ventilated place.",
            "Follow local heat-health advisories."
        ]
    },

    "Safe": {
        "level": "LOW",
        "message": [
            "No major risk detected from the supplied conditions.",
            "Continue monitoring official weather and disaster alerts.",
            "Keep a basic emergency plan ready."
        ]
    }
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])
        wind_speed = float(request.form["wind_speed"])
        water_level = float(request.form["water_level"])
        earthquake = float(request.form["earthquake"])
        slope = float(request.form["slope"])

        features = np.array([[
            rainfall,
            temperature,
            wind_speed,
            water_level,
            earthquake,
            slope
        ]])

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        classes = model.classes_

        probability = max(probabilities) * 100

        result = recommendations[prediction]

        return jsonify({
            "risk": prediction,
            "risk_level": result["level"],
            "probability": round(probability, 2),
            "recommendations": result["message"]
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )