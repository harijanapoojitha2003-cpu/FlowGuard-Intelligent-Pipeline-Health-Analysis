# ==========================================
# PIPELINE RISK PREDICTION PROJECT
# ==========================================

# -------- STEP 1: IMPORT LIBRARIES --------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# -------- STEP 2: LOAD DATASET --------
data = pd.read_csv("../dataset/water_leak_detection_20000_rows.csv")

print("Dataset Loaded Successfully")
print("Initial Shape:", data.shape)

# -------- STEP 3: PREPROCESSING --------
data = data.dropna()
data = data.drop_duplicates()
data = data.drop(columns=["Timestamp", "Sensor_ID"])
print("After Cleaning Shape:", data.shape)

# -------- STEP 4: ENCODE TARGET --------
le = LabelEncoder()
data["Risk_Level"] = le.fit_transform(data["Risk_Level"])

# -------- STEP 5: SPLIT FEATURES & TARGET --------
X = data.drop("Risk_Level", axis=1)
y = data["Risk_Level"]

# -------- STEP 6: SCALING --------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------- STEP 7: TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)

# ==========================================
# MODEL 1: RANDOM FOREST
# ==========================================
rf_model = RandomForestClassifier(
n_estimators=30,
max_depth=4,
random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

# ==========================================
# MODEL 2: LOGISTIC REGRESSION
# ==========================================
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)

# ==========================================
# BEST MODEL SELECTION
# ==========================================
if rf_accuracy > lr_accuracy:
    best_model = rf_model
    print("\nBest Model: Random Forest")
else:
    best_model = lr_model
    print("\nBest Model: Logistic Regression")

# -------- SAVE MODEL --------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel Saved Successfully!")

