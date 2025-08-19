import pandas as pd

input_file = "D:/EV_prediction/ev-charging-prediction/data/Aggregated_ev_data.csv"
output_file = "D:/EV_prediction/ev-charging-prediction/data/Cleaned_data.csv"

chunk_size = 500000

first_chunk = True

for chunk in pd.read_csv("D:/EV_prediction/ev-charging-prediction/data/Aggregated_ev_data.csv",chunksize = 500000 ):
    chunk['time'] = pd.to_datetime(chunk['time'])

    chunk.ffill(inplace = True)


    chunk['hour'] = chunk['time'].dt.hour
    chunk['weekday'] = chunk['time'].dt.day_name()
    chunk['month'] = chunk['time'].dt.month
    chunk['is_weekend'] = chunk['time'].dt.dayofweek >= 5

    if first_chunk:
        chunk.to_csv(output_file,index = False,mode = 'w')
        first_chunk = False
    else:
        chunk.to_csv(output_file,index  = False,mode = 'a',header = False)

first_chunk = True
for chunk in pd.read_csv("D:/EV_prediction/ev-charging-prediction/data/Cleaned_data.csv",chunksize  = 500000,on_bad_lines = 'skip'):
    chunk["avg_price"] = (chunk['s_price']+chunk['e_price'])/2
     
    if first_chunk:
        chunk.to_csv(output_file,index = False,mode = 'w')
        first_chunk = False
    else:
        chunk.to_csv(output_file,index  = False,mode = 'a',header = False)

df = pd.read_csv(output_file)
df.drop(['time','s_price','e_price'], axis = 1, inplace = True)
df.to_csv("D:/EV_prediction/ev-charging-prediction/data/Cleaned_data.csv",index = False)