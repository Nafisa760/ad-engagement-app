import streamlit as st
from inference import predict

st.set_page_config(page_title="Ad Engagement Predictor")

st.title("📊 Ad Engagement Prediction App")
st.write("Enter details below to predict engagement score:")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
gender = st.selectbox("Gender", ["M","F"])
location = st.selectbox("Location", ["USA","UK","India"])
interests = st.selectbox("Interests", ["Food","Tech","Gaming"])
ad_category = st.selectbox("Ad Category", ["Sportswear","Electronics","Luggage","Gadgets"])
ad_platform = st.selectbox("Ad Platform", ["Facebook","Instagram"])
ad_type = st.selectbox("Ad Type", ["Image","Video","Carousel"])
impressions = st.number_input("Impressions", min_value=0, value=10)
clicks = st.number_input("Clicks", min_value=0, value=4)
conversion = st.number_input("Conversion (0/1)", min_value=0, max_value=1, value=0)
time_spent = st.number_input("Time Spent on Ad (seconds)", min_value=0.0, value=13.2)
day_of_week = st.selectbox("Day of Week", 
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
device_type = st.selectbox("Device Type", ["Mobile","Tablet","Desktop"])

if st.button("Predict Engagement Score"):
    input_data = {
        "age": age,
        "gender": gender,
        "location": location,
        "interests": interests,
        "ad_category": ad_category,
        "ad_platform": ad_platform,
        "ad_type": ad_type,
        "impressions": impressions,
        "clicks": clicks,
        "conversion": conversion,
        "time_spent_on_ad": time_spent,
        "day_of_week": day_of_week,
        "device_type": device_type
    }
    result = predict(input_data)
    st.success(f"Predicted Engagement Score: {result:.4f}")
