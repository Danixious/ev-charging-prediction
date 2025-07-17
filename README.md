⚡ EV Charging Demand Prediction
Predicting future EV charging demand using Machine Learning to optimize resource planning and grid efficiency. This project leverages a Random Forest model trained on historical EV charging data, delivering high accuracy and practical insights for infrastructure scaling and energy distribution.

📊 Project Overview
Goal: Forecast charging volume (in kWh) based on temporal and contextual features.

Dataset: EV charging station usage logs including time, weekday, pricing, and volume.

Tech Stack: Python, Pandas, Scikit-learn, Matplotlib, XGBoost, Streamlit

🚀 Features
📈 Real-time prediction: Input data via Streamlit UI for immediate results.

🧾 Batch inference: Upload CSV files and get predictions for all records.

🔍 Exploratory Data Analysis: Understand charging patterns by day, hour, and price range.

📦 Deployed via Streamlit for interactive usage.

📌 Visual Insights
🔹 Average Charging Volume by Weekday
<img src="https://raw.githubusercontent.com/yourusername/your-repo/main/assets/weekday_volume.png" width="600"/>
🔹 Average Charging Volume by Hour
<img src="https://raw.githubusercontent.com/yourusername/your-repo/main/assets/hourly_volume.png" width="600"/>
🔹 Price vs Volume Distribution
<img src="https://raw.githubusercontent.com/yourusername/your-repo/main/assets/price_vs_volume.png" width="600"/>
🧠 Model Performance
Model	MSE	MAE	R² Score	Cross Validation
Linear Regression	0.0120	0.0788	0.9880	–
Random Forest	8.16×10⁻⁷	0.00011	0.999999	[0.99999891, 0.99999898, 0.99999885, 0.99999898, 0.99999845]
XGBoost	5.20×10⁻⁶	0.00075	0.999995	[ -0.54, 0.97, 0.99, -10.46, -2.20 ]
Mean: -2.25

✅ Best Model Chosen: Random Forest Regressor due to its highest accuracy and robust CV score.