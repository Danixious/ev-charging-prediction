# Import required libraries for data handling, ML model loading, UI, and visualization
import pandas as pd
import joblib
import numpy as np
import streamlit as st
import altair as alt
import os

# Get the absolute path of the current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the trained model file
MODEL_PATH = os.path.join(BASE_DIR, "../models/RandomForsetRegressorModel.joblib")

# Load the trained model and feature order used during training
model, feature_order = joblib.load(MODEL_PATH)

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Predict EV Charging Demand",
    layout="centered"
)

# Display the title of the application
st.title("⚡ EV Charging Demand Prediction")

# Sidebar option to select prediction mode
mode = st.sidebar.radio("Select Prediction Mode", ["Single Input", "Batch Upload"])


# Function to preprocess input data so it matches the model training format
def preprocess(df):

    # Define weekday order used during training
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Ensure weekday column exists
    if 'weekday' not in df.columns:
        df['weekday'] = 'Monday'

    # Convert weekday to categorical with correct order
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekdays)

    # Calculate average price if both price columns exist
    if 's_price' in df.columns and 'e_price' in df.columns:
        df['avg_price'] = (df['s_price'] + df['e_price']) / 2

    # One-hot encode weekday column
    df = pd.get_dummies(df, columns=['weekday'], drop_first=False)

    # Drop unnecessary columns that were removed during training
    df = df.drop(columns=[col for col in ['e_price', 's_price', 'time'] if col in df.columns], errors='ignore')

    # Ensure all model features exist in the input
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training feature order
    df = df[feature_order]

    return df


# ---------------- SINGLE INPUT PREDICTION ----------------
if mode == "Single Input":

    # Section title
    st.subheader("🔍 Predict for a Single Time Slot")

    # User inputs for model features
    busy = st.number_input("Busy Chargers", min_value=0.0, step=1.0)
    idle = st.number_input("Idle Chargers", min_value=0.0, step=1.0)
    fast_busy = st.number_input("Fast Chargers Busy", min_value=0.0)
    slow_busy = st.number_input("Slow Chargers Busy", min_value=0.0)
    duration = st.number_input("Charging Duration (hours)", min_value=0.0)
    weekday = st.selectbox("Weekday", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    hour = st.slider("Hour of Day", 0, 23)
    month = st.slider("Month", 1, 12)
    s_price = st.number_input("Starting Price", min_value=0.0)
    e_price = st.number_input("Ending Price", min_value=0.0)

    # Run prediction when button is clicked
    if st.button("Predict"):

        # Create dataframe from user input
        input_df = pd.DataFrame({
            'busy': [busy],
            'idle': [idle],
            'fast_busy': [fast_busy],
            'slow_busy': [slow_busy],
            'duration': [duration],
            'hour': [hour],
            'month': [month],
            'weekday': [weekday],
            's_price': [s_price],
            'e_price': [e_price]
        })

        # Preprocess input to match training format
        processed_df = preprocess(input_df)

        # Generate prediction using trained model
        prediction = model.predict(processed_df)

        # Calculate average price for display
        avg_price = (s_price + e_price) / 2

        # Display prediction result
        st.success(f"🔋 Estimated Energy Demand: {prediction[0]:.2f} kWh")

        # Show a styled summary of inputs
        st.markdown(f"""
        <div style='background-color:#262730; padding:15px; border-radius:10px; margin-top:20px;'>
        📘 <strong>Input Summary</strong><br><br>
        ⚡ Fast Charging Enabled: {"Yes" if fast_busy > 0 else "No"}<br>
        🕓 Hour of Day: {hour}:00<br>
        💰 Average Price: ₹{avg_price:.2f}/kWh<br>
        ⛽ Busy Slots: {busy}<br>
        🚗 Idle Slots: {idle}
        </div>
        """, unsafe_allow_html=True)


# ---------------- BATCH CSV PREDICTION ----------------
elif mode == "Batch Upload":

    # Section title
    st.subheader("📂 Predict from CSV Upload")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:

        # Read uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Show preview of uploaded data
        st.write("Uploaded Data Preview:", df.head())

        # Preprocess input data
        processed_df = preprocess(df)

        # Generate predictions
        predictions = model.predict(processed_df)

        # Add predictions to dataframe
        df["Predicted Volume"] = predictions

        # Display predictions
        st.write("🔋 Predictions", predictions)

        # ---------------- VISUALIZATION ----------------

        # Plot predicted demand by hour
        if 'hour' in df.columns:
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('hour:Q', title='Hour of Day'),
                y=alt.Y('Predicted Volume:Q', title='Predicted Charging Volume'),
                tooltip=['hour','Predicted Volume']
            ).properties(title='Predicted Charging Demand by Hour')

            st.altair_chart(chart, use_container_width=True)

        # Plot average predicted demand per weekday
        if 'weekday' in df.columns:

            weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)

            avg_by_day = df.groupby('weekday')['Predicted Volume'].mean().reset_index()

            chart = alt.Chart(avg_by_day).mark_bar().encode(
                x='weekday',
                y='Predicted Volume',
                tooltip=['weekday','Predicted Volume']
            ).properties(title="Average Predicted Demand per Weekday")

            st.altair_chart(chart, use_container_width=True)

        # Create histogram distribution of predicted volume
        hist_data = df['Predicted Volume'].round(1).value_counts().reset_index()
        hist_data.columns = ['Volume','Count']

        chart = alt.Chart(hist_data).mark_bar().encode(
            x='Volume',
            y='Count',
            tooltip=['Volume','Count']
        ).properties(title="Distribution of Predicted Charging Volume")

        st.altair_chart(chart, use_container_width=True)

        # Allow user to download predictions
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime='text/csv'
        )