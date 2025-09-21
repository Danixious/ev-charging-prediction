import pandas as pd
import joblib
import numpy as np
import streamlit as st
import altair as alt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import threading
import requests
import io
import os

API_URL = os.getenv("API_URL", "http://api:8000")
model, feature_order = joblib.load("../models/RandomForsetRegressorModel.joblib")

st.set_page_config(
    page_title="Predict the charging volume based on station conditions and time",
    layout="centered"
)

st.title("⚡EV Charging Demand Prediction")

mode = st.sidebar.radio("Select Prediction Mode", ["Single Input", "Batch Upload"])

def preprocess(df):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if 'weekday' not in df.columns:
        if 'day_of_week' in df.columns:
            df['weekday'] = df['day_of_week'].apply(lambda x: weekdays[int(x)] if 0 <= int(x) <= 6 else 'Unknown')
        else:
            df['weekday'] = 'Monday'
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekdays)
    if 's_price' in df.columns and 'e_price' in df.columns:
        df['avg_price'] = (df['s_price'] + df['e_price']) / 2
    df = pd.get_dummies(df, columns=['weekday'], drop_first=False)
    df = df.drop(columns=[col for col in ['e_price', 's_price', 'time'] if col in df.columns], errors='ignore')
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_order]
    return df

# ---------------- SINGLE INPUT ----------------
if mode == "Single Input":
    st.subheader("🔍Predict for a Single Time Slot")

    busy = st.number_input("Busy Chargers", min_value=0.0, step=1.0)
    idle = st.number_input("Idle Chargers", min_value=0.0, step=1.0)
    fast_busy = st.number_input("Fast Chargers Busy", min_value=0.0)
    slow_busy = st.number_input("Slow Chargers Busy", min_value=0.0)
    duration = st.number_input("Charging Duration (hours)", min_value=0.0)
    weekday = st.selectbox("Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hour = st.slider("Hour of Day", 0, 23)
    month = st.slider("Month", 1, 12)
    s_price = st.number_input("Starting Price", min_value=0.0)
    e_price = st.number_input("Ending Price", min_value=0.0)

    if st.button("Predict"):
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

        input_df['avg_price'] = (input_df['s_price'] + input_df['e_price']) / 2

        try:
            response = requests.post(
                f"{API_URL}/predict_single/",
                json={"features": input_df.iloc[0].to_dict()}
            )
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                avg_price = input_df['avg_price'].iloc[0]

                st.success(f"🔋 Estimated Energy Demand: {prediction[0]:.2f} kWh")

                summary_html = f"""
                <div style='background-color:#262730; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                    <span style='font-size:18px'>📘 <strong>Input Summary:</strong></span> <br>
                    <span style='color:#DDDDDD;'>
                        ⚡ <strong>Fast Charging Enabled:</strong> {"Yes" if fast_busy > 0 else "No"}<br>
                        🕓 <strong>Hour of Day:</strong> {hour}:00<br>
                        💰 <strong>Average Price:</strong> ₹{avg_price:.2f}/kWh<br>
                        ⛽ <strong>Busy Slots:</strong> {busy}<br>
                        🚗 <strong>Idle Slots:</strong> {idle}
                    </span>
                </div>
                """
                st.markdown(summary_html, unsafe_allow_html=True)
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"API connection failed: {e}")

# ---------------- BATCH UPLOAD ----------------
elif mode == "Batch Upload":
    st.subheader("📂 Predict from CSV Upload")

    uploaded_file = st.file_uploader("Upload a CSV file with data", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:", df.head())

        if 's_price' in df.columns and 'e_price' in df.columns:
            df['avg_price'] = (df['s_price'] + df['e_price']) / 2

        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{API_URL}/predict_batch/", files=files)

            if response.status_code == 200:
                result = response.json()
                predictions = result["predictions"]
                df["Predicted Volume"] = predictions
                st.write("🔋 Predictions", predictions)

                # --- Summary + Charts (your original code kept as-is) ---
                if 'Predicted Volume' in df.columns:
                    num_fast = df['fast_busy'].sum() if 'fast_busy' in df.columns else np.nan
                    avg_price = df['avg_price'].mean() if 'avg_price' in df.columns else np.nan
                    hours = df['hour'].nunique() if 'hour' in df.columns else np.nan
                    min_hour = df['hour'].min() if 'hour' in df.columns else np.nan
                    max_hour = df['hour'].max() if 'hour' in df.columns else np.nan
                    total_rows = len(df)
                    peak_hour = None
                    if 'hour' in df.columns:
                        peak_hour = df.groupby('hour')['Predicted Volume'].mean().idxmax()

                    st.markdown(f"""
                    <div style='background-color:#262730; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                        <span style='font-size:18px'>📊 <strong>Batch Upload Summary:</strong></span> <br>
                        <span style='color:#DDDDDD;'>
                            ✅ <strong>Total Records:</strong> {total_rows}<br>
                            ⚡ <strong>Fast Chargers:</strong> {num_fast}<br>
                            🕓 <strong>Time Range:</strong> {min_hour}:00 - {max_hour}:00 ({hours} unique hour(s))<br>
                            💰 <strong>Average Price:</strong> ₹{avg_price:.2f}/kWh<br>
                            🌟 <strong>Predicted Peak Demand Hour:</strong> {peak_hour if peak_hour is not None else '-'}:00
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    if 'hour' in df.columns:
                        chart = alt.Chart(df).mark_line(point=True).encode(
                            x=alt.X('hour:Q', title='Hour of Day'),
                            y=alt.Y('Predicted Volume:Q', title='Predicted Charging Volume'),
                            tooltip=['hour', 'Predicted Volume']
                        ).properties(title='🔁 Predicted Volume by Hour')
                        st.altair_chart(chart, use_container_width=True)

                    if 'weekday' in df.columns:
                        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
                        avg_by_day = df.groupby('weekday')['Predicted Volume'].mean().reset_index()

                        chart = alt.Chart(avg_by_day).mark_bar().encode(
                            x=alt.X('weekday:N', title='Day of the Week'),
                            y=alt.Y('Predicted Volume:Q', title='Avg Charging Volume'),
                            tooltip=['weekday', 'Predicted Volume']
                        ).properties(title='📊 Average Predicted Volume per Weekday')
                        st.altair_chart(chart, use_container_width=True)

                    hist_data = df['Predicted Volume'].round(1).value_counts().reset_index()
                    hist_data.columns = ['Volume', 'Count']

                    chart = alt.Chart(hist_data).mark_bar().encode(
                        x=alt.X('Volume:Q', title='Predicted Volume (rounded)'),
                        y=alt.Y('Count:Q', title='Frequency'),
                        tooltip=['Volume', 'Count']
                    ).properties(title='📈 Distribution of Predicted Volume')
                    st.altair_chart(chart, use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime='text/csv')
            else:
                st.error(f"API Error: {response.json().get('detail','Unknown error')}")
        except Exception as e:
            st.error(f"API connection failed: {e}")



