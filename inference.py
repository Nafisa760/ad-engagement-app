import joblib
import numpy as np
import os


base_path = os.path.dirname(os.path.abspath(__file__)) + "/"



model = joblib.load(base_path + "best_rf.joblib")
scaler = joblib.load(base_path + "scaler.joblib")
label_encoders = joblib.load(base_path + "label_encoders.joblib")

FEATURE_ORDER = [
    "age", "gender", "location", "interests", "ad_category",
    "ad_platform", "ad_type", "impressions", "clicks",
    "conversion", "time_spent_on_ad", "day_of_week", "device_type"
]

def preprocess_input(input_dict):
    row = []
    for col in FEATURE_ORDER:
        val = input_dict[col]
        if col in label_encoders:
            le = label_encoders[col]
            if val in le.classes_.tolist():
                encoded = int(le.transform([val])[0])
            else:
                encoded = 0
            row.append(encoded)
        else:
            row.append(float(val))

    X = np.array(row).reshape(1, -1)
    X_scaled = scaler.transform(X)
    return X_scaled

def predict(input_dict):
    try:
        X_pre = preprocess_input(input_dict)
        pred = model.predict(X_pre)
        return float(pred[0])
    except Exception as e:
        print("Error in prediction:", e)
        return 0.0
