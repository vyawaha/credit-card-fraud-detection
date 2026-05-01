import pandas as pd

from src.predict import predict_transaction
from src.features import create_features

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Apply SAME feature engineering used during training
df = create_features(df)

# Remove target column
df = df.drop("Class", axis=1)

def simulate_stream():

    print("\n🚨 Real-Time Fraud Detection Simulation Started...\n")

    while True:

        # Pick random transaction
        sample = df.sample(1).iloc[0]

        # Convert to dictionary
        tx = sample.to_dict()

        # Predict
        result = predict_transaction(tx)

        print("=" * 60)
        print(f"💳 Amount: ₹{tx['Amount']:.2f}")
        print(f"⚠ Fraud Prediction: {result['fraud_prediction']}")
        print(f"📊 Fraud Probability: {result['fraud_probability']}")

        user_input = input(
            "\nPress ENTER for next transaction or q to quit: "
        )

        if user_input.lower() == "q":
            break

simulate_stream()