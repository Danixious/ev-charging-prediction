import pandas as pd
import sqlite3


df = pd.read_csv("D:/EV_prediction/ev-charging-prediction/data/Cleaned_data.zip")


conn = sqlite3.connect("D:/EV_prediction/ev-charging-prediction/notebooks/ev.db")
df.to_sql("ev_data", conn, if_exists="replace", index=False)
conn.close()

print("✅ Data successfully loaded into ev.db")

