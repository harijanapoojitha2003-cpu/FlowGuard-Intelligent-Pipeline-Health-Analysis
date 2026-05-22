# STEP 3: Data Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1️⃣ Load Dataset
data = pd.read_csv("water_leak_detection_20000_rows.csv")

print("✅ Dataset Loaded Successfully")
print("Total Rows:", data.shape[0])
print("Total Columns:", data.shape[1])

# 2️⃣ Drop unnecessary columns
data = data.drop(["Timestamp", "Sensor_ID"], axis=1)

# 3️⃣ Separate Features and Target
X = data.drop("Risk_Level", axis=1)
y = data["Risk_Level"]

# 4️⃣ Encode Target (Low=0, Medium=1, High=2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 5️⃣ Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain Size:", X_train.shape[0])
print("Test Size:", X_test.shape[0])

# 6️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n✅ Preprocessing Completed Successfully!")