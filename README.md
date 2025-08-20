# ⚡ EV Charging Demand Prediction
>Predict future EV charging demand using Machine Learning to optimize grid efficiency and infrastructure planning.  
> 🧠 Powered by a Random Forest model trained on real usage data.

---

## Live Demo
- 👉 [EV Charging Demand Prediction Web App](https://ev-charging-prediction-ca4n5rs8xzdrwwsr8buin8.streamlit.app/)

---

- 📄 [Data Used in Training ](https://github.com/Danixious/ev-charging-prediction/blob/main/data/Cleaned_data.zip)


## 📊 Project Overview
-  🎯 **Goal:** Forecast charging volume (in kWh) based on temporal and contextual features.

- 🗂 **Dataset:** EV charging station usage logs including time, weekday, pricing, and volume.

- 🧰 **Tech Stack:**
  ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
  ![XGBoost](https://img.shields.io/badge/XGBoost-EC6C00?style=flat)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

---

## 🚀 Features
- 📈 **Real-Time Prediction:** Input data through the Streamlit UI  

- 🧾 **Batch Inference:** Upload CSVs for multi-record predictions  

- 🔍 **EDA:** Explore demand patterns by day, hour, and pricing  

- 💻 **Streamlit App:** Deployed for interactive user experience  

---

## 📥 Input Variables Explained

| Feature            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `hour`             | Hour of the day (0–23)                                                      |
| `s_price`          | Starting price for EV charging (₹/kWh)                                      |
| `e_price`          | Ending price for EV charging (₹/kWh)                                        |
| `avg_price`        | Average of `s_price` and `e_price` (automatically calculated by the app)    |
| `slow_busy`        | Number of busy slow chargers at the station                                 |
| `slow_idle`        | Number of idle slow chargers at the station                                 |
| `fast_busy`        | Number of busy fast chargers at the station                                 |
| `fast_idle`        | Number of idle fast chargers at the station                                 |

---

## 🧠 Model Performance
| Model             | MSE         | MAE      | R² Score   | Cross Validation Scores                        |
|------------------|-------------|----------|------------|------------------------------------------------|
| Linear Regression| 0.0120      | 0.0788   | 0.9880     | –                                              |
| Random Forest    | 8.16×10⁻⁷   | 0.00011  | 0.999999   | [0.99999891, 0.99999898, ..., 0.99999845]      |
| XGBoost          | 5.20×10⁻⁶   | 0.00075  | 0.999995   | [-0.54, 0.97, 0.99, -10.46, -2.20] <br>**Mean**: -2.25 |


✅ **Best Model Chosen:** Random Forest Regressor due to its highest accuracy and robust CV score.

---

## 🖥️ App Preview

<img src="https://github.com/Danixious/ev-charging-prediction/blob/main/outputs/AppPreview.png" width="700"/>

---

## 🧪 How to Run

-  Clone the repo:
git clone https://github.com/Danixious/ev-charging-prediction.git
cd ev-charging-prediction

-  Install dependencies:
pip install -r requirements.txt

- Run Streamlit App:
streamlit run app.py


## 📁 Project Structure

├── app.py        # Streamlit UI
                             
├── model/        # Trained model (Random Forest)
                            
├── notebooks/    # EDA and model training
                             
├── data/         # Cleaned dataset
                  
├── outputs/      # Visualizations
                               
├── README.md

└── requirements.txt

## ✨ Future Improvements
- Integrate weather & traffic data

- Deploy on cloud (AWS/GCP)

- Add time-series models (Prophet, LSTM)

- Feedback mechanism for live model retraining

## 🙋‍♂️ Author
Daniel Julius Natal

Computer Science Engineering student

📧 Email(mailto:danieljuliusnatal@gmail.com) | 💼 LinkedIn(www.linkedin.com/in/daniel-julius-natal-68060228a)

