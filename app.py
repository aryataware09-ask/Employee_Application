import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Employee Prediction App", page_icon="📊", layout="centered"
)

# Custom CSS for shadow effects and clean layout
st.markdown(
    """
    <style>
    .main-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the model with caching
@st.cache_resource
def load_model():
    with open("naive.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# App Header
st.title("📊 Naive Bayes Prediction App")
st.write(
    "Provide the employee details below to get a prediction from the trained model."
)

st.markdown("---")

# Input Form Layout
with st.form("prediction_form"):
    st.subheader("Employee Details Input")

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Education", ["Bachelors", "Masters", "PHD"]
        )  # Adjust categories as per your training data
        joining_year = st.number_input(
            "Joining Year", min_value=2000, max_value=2030, value=2015, step=1
        )
        city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
        payment_tier = st.selectbox(
            "Payment Tier", [1, 2, 3]
        )  # Usually tiers 1, 2, 3

    with col2:
        age = st.slider("Age", min_value=18, max_value=65, value=28)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ever_benched = st.selectbox("Ever Benched", ["No", "Yes"])
        experience_in_current_domain = st.slider(
            "Experience In Current Domain (Years)",
            min_value=0,
            max_value=15,
            value=2,
        )

    submit_button = st.form_submit_button(label="Predict")

# Prediction logic
if submit_button:
    # NOTE: Depending on how your categorical variables (Education, City, Gender, EverBenched)
    # were encoded during training (e.g., OneHotEncoding, LabelEncoding), you must match that encoding here.
    # Below is a sample mapping structure. Update it if your pipeline or preprocessing logic differs.

    # Example categorical mapping placeholder (modify according to your exact training preprocessing):
    edu_mapping = {"Bachelors": 0, "Masters": 1, "PHD": 2}
    city_mapping = {"Bangalore": 0, "Pune": 1, "New Delhi": 2}
    gender_mapping = {"Male": 0, "Female": 1}
    benched_mapping = {"No": 0, "Yes": 1}

    # Construct the feature array in the exact order expected by feature_names_in_
    # Features: ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']
    features = np.array(
        [[
            edu_mapping.get(education, 0),
            joining_year,
            city_mapping.get(city, 0),
            payment_tier,
            age,
            gender_mapping.get(gender, 0),
            benched_mapping.get(ever_benched, 0),
            experience_in_current_domain,
        ]]
    )

    try:
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        st.markdown("---")
        st.subheader("Prediction Result")
        st.success(f"The predicted class is: **{prediction[0]}**")

        st.write(
            "**Prediction Probabilities:**",
            pd.DataFrame(
                prediction_proba, columns=model.classes_
            ).style.format("{:.2%}"),
        )
    except Exception as e:
        st.error(
            "Error making prediction. Note: If your model expects string categories or One-Hot Encoded columns instead of numeric mappings, update the input array mapping accordingly."
        )
        st.error(f"Details: {e}")
