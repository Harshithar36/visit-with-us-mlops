
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Visit With Us - Package Purchase Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Visit With Us")
st.subheader("Tourism Package Purchase Prediction")

st.write(
    "Enter customer details to predict whether the customer is likely "
    "to purchase the tourism package."
)

# Numerical inputs
age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=0.0,
    value=10.0
)
number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)
number_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0.0,
    value=3.0
)
preferred_property_star = st.number_input(
    "Preferred Property Star",
    min_value=1.0,
    max_value=5.0,
    value=3.0
)
number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0.0,
    value=3.0
)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)
own_car = st.selectbox("Own Car", [0, 1])
number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0.0,
    value=1.0
)
monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)

# Categorical inputs
type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Fe Male"]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"]
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

# Create input DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})

st.write("### Customer Input")
st.dataframe(input_data)

# Prediction
if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(
            f"Customer is likely to purchase the package. "
            f"Probability: {probability:.2%}"
        )
    else:
        st.info(
            f"Customer is unlikely to purchase the package. "
            f"Probability: {probability:.2%}"
        )
