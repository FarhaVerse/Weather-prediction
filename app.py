
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Weather Prediction",
    page_icon="🌦️",
    layout="centered"
)

model = joblib.load("weather_prediction_model.pkl")
scaler = joblib.load("weather_scaler.pkl")
encoders = joblib.load("weather_encoders.pkl")

st.title("🌦️ Weather Prediction System")
st.write("Predict weather using Machine Learning")

st.divider()

st.subheader("Enter Weather Details")

weather_inputs = {}

for column in model.feature_names_in_:

    if column in encoders:

        options = list(encoders[column].classes_)

        weather_inputs[column] = st.selectbox(
            column,
            options
        )

    else:

        weather_inputs[column] = st.number_input(
            column,
            value=25.0
        )

if st.button("Predict Weather", use_container_width=True):

    input_df = pd.DataFrame([weather_inputs])

    for column in input_df.columns:

        if column in encoders:
            input_df[column] = encoders[column].transform(
                input_df[column]
            )

    prediction = model.predict(input_df)

    result = encoders['Weather Type'].inverse_transform(
        prediction
    )[0]

    st.divider()

    st.success("Weather Prediction Completed!")

    st.subheader("Predicted Weather")

    if result == "Sunny":
        st.markdown("# ☀️ Sunny")

    elif result == "Rainy":
        st.markdown("# 🌧️ Rainy")

    elif result == "Cloudy":
        st.markdown("# ☁️ Cloudy")

    elif result == "Snowy":
        st.markdown("# ❄️ Snowy")

    else:
        st.markdown(f"# 🌤️ {result}")

    st.balloons()
