import pandas as pd
import joblib
import numpy
import streamlit as st
import altair as alt

model,feature_order = joblib.load("models/RandomForsetRegressorModel.joblib")
st.set_page_config(page_title = "Predict the charging volume based on station conditions and time",layout = "centered")
st.title("⚡EV Charging Demand Prediction")


mode = st.sidebar.radio("Select Prediction Mode",["Single Input","Batch Upload"])

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

if mode == "Single Input":
    st.subheader("🔍Predict for a Single Time Slot")

    busy = st.number_input("Busy Chargers",min_value = 0.0,step = 1.0)
    idle = st.number_input("Idle Chargers", min_value=0.0,step = 1.0)
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
        processed = preprocess(input_df)
        processed = processed.reindex(columns = feature_order,fill_value=0)
        prediction = model.predict(processed)
        st.success(f"🔋 Estimated Energy Demand: {prediction[0]:.2f} kWh")
        st.markdown(
           f"""<div style='padding: 10px; background-color: #f1f3f4; border-radius: 6px;'>
           <strong>🔍 Interpretation:</strong> Based on the current station status and time conditions, the estimated energy demand for EV charging is approximately <strong>{prediction[0]:.2f} kilowatt-hours (kWh)</strong> during this timeslot. This value can help anticipate electricity load and manage station capacity.
           </div>""",
           unsafe_allow_html=True
    )
elif mode == "Batch Upload":
    st.subheader("📂 Predict from CSV Upload")

    uploaded_file = st.file_uploader("Upload a CSV file with data", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:", df.head())

        if 's_price' in df.columns and 'e_price' in df.columns:
           df['avg_price'] = (df['s_price'] + df['e_price']) / 2

        processed = preprocess(df)
        processed = processed.reindex(columns = feature_order,fill_value=0)
        predictions = model.predict(processed)

        df['Predicted Volume'] = predictions
        st.write("🔋 Predictions", df.head())
        st.markdown(
           f"""<div style='padding: 10px; background-color: #f1f3f4; border-radius: 6px;'>
           <strong>📊 Interpretation:</strong> The uploaded dataset has been successfully processed. Each row now includes a <strong>Predicted Charging Volume</strong> (in kWh), which estimates the expected energy demand based on station status, time, and pricing data.
           </div>""",
        unsafe_allow_html=True
)

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
