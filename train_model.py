import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

data = []

# Generate synthetic training data
for _ in range(5000):

    rainfall = np.random.uniform(0, 300)
    temperature = np.random.uniform(5, 50)
    wind_speed = np.random.uniform(0, 180)
    water_level = np.random.uniform(0, 15)
    earthquake = np.random.uniform(0, 9)
    slope = np.random.uniform(0, 60)

    # Risk rules used to generate training examples
    if earthquake >= 6:
        risk = "Earthquake"

    elif wind_speed >= 100:
        risk = "Cyclone"

    elif rainfall >= 180 and water_level >= 8:
        risk = "Flood"

    elif rainfall >= 120 and slope >= 35:
        risk = "Landslide"

    elif temperature >= 42:
        risk = "Heatwave"

    else:
        risk = "Safe"

    data.append([
        rainfall,
        temperature,
        wind_speed,
        water_level,
        earthquake,
        slope,
        risk
    ])

columns = [
    "rainfall",
    "temperature",
    "wind_speed",
    "water_level",
    "earthquake",
    "slope",
    "risk"
]

df = pd.DataFrame(data, columns=columns)

X = df.drop("risk", axis=1)
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "disaster_model.pkl")

print("AI model saved as disaster_model.pkl")