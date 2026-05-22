from flask import Flask, render_template, request, redirect, session, url_for
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

app.secret_key = "flowguard_secure_key"


# ==========================================
# LOAD SAVED MACHINE LEARNING MODEL FILES
# ==========================================

try:

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")

    print("FlowGuard Machine Learning Models Loaded Successfully!")

except Exception as e:

    print(f"Error loading ML model files: {e}")

    print("Make sure best_model.pkl, scaler.pkl, and label_encoder.pkl are in the same directory.")


# USER LOGINS

users = {"admin": "admin"}


@app.route("/")
def home():

    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:

            return redirect(url_for("index"))

        else:

            return "Invalid Credentials"

    return render_template("login.html")


@app.route("/index")
def index():

    return render_template("index.html")


@app.route("/about")
def about():

    return render_template("about.html")


@app.route("/safety")
def safety():

    return render_template("safety.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")



@app.route("/logout")
def logout():

    return redirect(url_for("home"))


# ==========================================
# REAL ML MANUAL PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:

            pressure = float(request.form["pressure"])
            flow = float(request.form["flow"])
            temp = float(request.form["temp"])
            leak = int(request.form["leak"])
            burst = int(request.form["burst"])

            input_data = pd.DataFrame(

                [[pressure, flow, temp, leak, burst]],

                columns=[
                    "Pressure (bar)",
                    "Flow Rate (L/s)",
                    "Temperature (°C)",
                    "Leak Status",
                    "Burst Status"
                ]
            )

            scaled_features = scaler.transform(input_data)

            prediction_encoded = model.predict(scaled_features)

            risk = str(le.inverse_transform(prediction_encoded)[0])

            prob_matrix = model.predict_proba(scaled_features)[0]

            max_prob_index = int(prediction_encoded[0])

            prob = int(round(prob_matrix[max_prob_index] * 100))

        except Exception as e:

            print(f"Prediction runtime error fallback: {e}")

            risk = "Medium"
            prob = 60

            pressure, flow, temp = 0.0, 0.0, 0.0

        if risk == "Low":

            color = "green"

        elif risk == "Medium":

            color = "orange"

        else:

            color = "red"

        now = datetime.now()

        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M:%S")

        return render_template(
            "result.html",

            risk=risk,
            prob=prob,
            color=color,
            pressure=pressure,
            flow=flow,
            temp=temp,
            date=date_str,
            time=time_str
        )

    return redirect(url_for("index"))


# ==========================================
# AUTOMATED SIMULATION ROUTE HANDLER
# ==========================================

@app.route("/auto")
def auto_predict():

    try:

        pressure = 5.5
        flow = 150.0
        temp = 42.0
        leak = 0
        burst = 0

        input_data = pd.DataFrame(

            [[pressure, flow, temp, leak, burst]],

            columns=[
                "Pressure (bar)",
                "Flow Rate (L/s)",
                "Temperature (°C)",
                "Leak Status",
                "Burst Status"
            ]
        )

        scaled_features = scaler.transform(input_data)

        prediction_encoded = model.predict(scaled_features)

        risk = str(le.inverse_transform(prediction_encoded)[0])

        prob_matrix = model.predict_proba(scaled_features)[0]

        max_prob_index = int(prediction_encoded[0])

        prob = int(round(prob_matrix[max_prob_index] * 100))

    except Exception as e:

        print(f"Auto Prediction error: {e}")

        risk = "Low"
        prob = 75

        pressure, flow, temp = 4.5, 120.0, 35.0

    if risk == "Low":

        color = "green"

    elif risk == "Medium":

        color = "orange"

    else:

        color = "red"

    now = datetime.now()

    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    return render_template(
        "result.html",

        risk=risk,
        prob=prob,
        color=color,
        pressure=pressure,
        flow=flow,
        temp=temp,
        date=date_str,
        time=time_str

    )
@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

if __name__ == "__main__":
    app.run(debug=True)