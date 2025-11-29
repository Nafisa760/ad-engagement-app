import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1️⃣ Load your dataset
df = pd.read_csv("social_media_ad_optimization.csv")  # تأكد أن الملف موجود في نفس المجلد

FEATURE_ORDER = [
    "age", "gender", "location", "interests", "ad_category",
    "ad_platform", "ad_type", "impressions", "clicks",
    "conversion", "time_spent_on_ad", "day_of_week", "device_type"
]

target = "engagement_score"

# 2️⃣ Preprocess data
label_encoders = {}
X = df[FEATURE_ORDER].copy()
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = df[target].values

# 3️⃣ Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 4️⃣ Save artifacts to a specific folder
save_path = r"C:\Users\alzah\applied\\"   
os.makedirs(save_path, exist_ok=True)

joblib.dump(model, save_path + "best_rf.joblib")
joblib.dump(scaler, save_path + "scaler.joblib")
joblib.dump(label_encoders, save_path + "label_encoders.joblib")

# 5️⃣ Verify files were saved
for f in ["best_rf.joblib", "scaler.joblib", "label_encoders.joblib"]:
    print(f, os.path.exists(save_path + f))

print("Model, scaler, and label_encoders saved successfully in:", save_path)
