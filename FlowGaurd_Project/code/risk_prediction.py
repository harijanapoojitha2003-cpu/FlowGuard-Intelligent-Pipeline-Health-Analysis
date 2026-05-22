# ==========================================
# RISK PREDICTION USING SAVED MODEL
# ==========================================

import joblib
import pandas as pd

# -------- LOAD SAVED FILES --------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")


# -------- NEW INPUT DATA --------
# Example sensor values
new_data = pd.DataFrame(
    [[4.5, 12.0, 35.0, 1, 0]],
    columns=[
        "Pressure (bar)",
        "Flow Rate (L/s)",
        "Temperature (°C)",
        "Leak Status",
        "Burst Status"
    ]
)


# -------- SCALE DATA --------
scaled_data = scaler.transform(new_data)


# -------- MAKE PREDICTION --------
prediction = model.predict(scaled_data)


# -------- CONVERT BACK TO LABEL --------
risk_level = le.inverse_transform(prediction)


# -------- PRINT RESULT --------
print("\nPredicted Risk Level:", risk_level[0])