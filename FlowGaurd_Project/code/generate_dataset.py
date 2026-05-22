import pandas as pd
import numpy as np
from datetime import datetime, timedelta

num_records = 20000

# Timestamp format same as yours (dd-mm-yyyy HH:MM)
start_time = datetime(2024, 1, 1, 0, 0)
timestamps = [(start_time + timedelta(minutes=5*i)).strftime("%d-%m-%Y %H:%M") 
              for i in range(num_records)]

sensor_ids = np.random.choice(
    ["S001","S002","S003","S004","S005","S006","S007","S008","S009","S010"],
    num_records
)

pressure = np.round(np.random.uniform(2.0, 4.5, num_records), 9)
flow_rate = np.round(np.random.uniform(50, 220, num_records), 9)
temperature = np.round(np.random.uniform(10, 30, num_records), 9)

leak_status = np.random.choice([0, 1], num_records)
burst_status = np.random.choice([0, 1], num_records)

risk = []

for i in range(num_records):
    if pressure[i] > 4.2 and flow_rate[i] > 180:
        risk.append("High")

    elif leak_status[i] == 1 and pressure[i] > 3.5:
        risk.append("Medium")

    elif temperature[i] > 28:
        risk.append("Medium")

    else:
        risk.append("Low")

df = pd.DataFrame({
    "Timestamp": timestamps,
    "Sensor_ID": sensor_ids,
    "Pressure (bar)": pressure,
    "Flow Rate (L/s)": flow_rate,
    "Temperature (°C)": temperature,
    "Leak Status": leak_status,
    "Burst Status": burst_status,
    "Risk_Level": risk
})

df.to_csv("water_leak_detection_20000_rows.csv", index=False)

print("✅ 20,000 records generated successfully!")