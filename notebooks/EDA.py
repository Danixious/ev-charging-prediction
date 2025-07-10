import pandas as pd
from datetime import datetime,date,time
import matplotlib.pyplot as plt
import seaborn as sns
for chunk in pd.read_csv("D:/EV_prediction/ev-charging-prediction/data/Aggregated_ev_data.csv",chunksize = 500000 ):
    print(chunk.head())
    print(chunk.info())
    print(chunk.describe())
    break

for chunk in pd.read_csv("D:/EV_prediction/ev-charging-prediction/data/Aggregated_ev_data.csv",chunksize = 500000 ):
    chunk['time'] = pd.to_datetime(chunk['time'])

    chunk['hour'] = chunk['time'].dt.hour
    chunk['weekday'] = chunk['time'].dt.day_name()
    chunk['month'] = chunk['time'].dt.month
    chunk['is_weekend'] = chunk['time'].dt.dayofweek >= 5

    print(chunk[['time','hour','weekday','month','is_weekend']].head())
    break

avg_hour = chunk.groupby('hour')['volume'].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.bar(avg_hour['hour'], avg_hour['volume'], color='skyblue')
plt.title("Average Charging Volume by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Volume (kWh)")
plt.xticks(range(0, 24))  
plt.grid(axis='y')
plt.tight_layout()
plt.show()


weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
chunk['weekday'] = pd.Categorical(chunk['weekday'], categories=weekday_order, ordered=True)


weekday_avg = chunk.groupby('weekday')['volume'].mean().reset_index()


plt.figure(figsize=(10, 5))
sns.barplot(data=weekday_avg, x='weekday', y='volume', palette='Oranges')
plt.title("Average Charging Volume by Weekday", fontsize=14)
plt.xlabel("Day of the Week")
plt.ylabel("Average Volume (kWh)")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

e_price_vs_volume = chunk.groupby('e_price')['volume'].mean().reset_index()

plt.figure(figsize =(10,5))
plt.bar(e_price_vs_volume['e_price'],e_price_vs_volume['volume'],color = 'Green')
plt.title("e_Price VS Volume")
plt.xlabel("Price")
plt.ylabel("Volume")
plt.xticks(range(0,3))
plt.grid(axis='y')
plt.tight_layout()
plt.show()

s_price_vs_volume = chunk.groupby('e_price')['volume'].mean().reset_index()

plt.figure(figsize =(10,5))
sns.barplot(data = s_price_vs_volume,x='e_price',y = 'volume',palette= 'Oranges')
plt.title("s_Price VS Volume")
plt.xlabel("Price")
plt.ylabel("Volume")
plt.xticks(range(0,3))
plt.grid(axis='y')
plt.tight_layout()
plt.show()

