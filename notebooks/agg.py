import pandas as pd
from pathlib import Path

path2csv = Path("D:/EV_prediction/ev-charging-prediction/data/UrbanEVDataset/UrbanEVDataset/20220901-20230228_station-raw/charge_5min")
csvlist = list(path2csv.glob("*.csv"))

required_columns = (['time','volume','busy','idle','fast_busy','slow_busy','s_price','e_price','duration'])
def check_column(csvlist,column_name):
    try:
        for files in csvlist:
            df  = pd.read_csv(files)
            if column_name not in df.columns:
                print(f"Missing column '{column_name}' in: {files.name}")
                return False
        return True
    except FileNotFoundError:
        print(f"Error: File '{csvlist}'not found")
        return False
    
column_name = "time"   
if check_column(csvlist,column_name):
    print("ok report")

    dfs = []
    for file in csvlist:
        try:
            df = pd.read_csv(file)
            df = df[required_columns]
            dfs.append(df)
        except Exception as e:
            print(f"Error parsing {file.name}: {e}")
    full_df = pd.concat(dfs, ignore_index=True)
    print("Combined DataFrame shape: ",full_df.shape)
else: 
    print("aborting.....")

aggregated_df = full_df.groupby('time').agg({
    'volume': 'sum',
    'busy': 'sum',
    'idle': 'sum',
    'fast_busy': 'sum',
    'slow_busy': 'sum',
    's_price': 'mean',
    'e_price': 'mean',
    'duration': 'mean'
}).reset_index()

full_df.to_csv("Aggregated_ev_data.csv",index = False)