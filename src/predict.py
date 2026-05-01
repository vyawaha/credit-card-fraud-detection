import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/fraud_model.pkl")

def predict_transaction(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 4)
    }