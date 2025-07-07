import pandas as pd
from pathlib import Path

path2csv = Path("D:/EV_prediction/ev-charging-prediction/data/UrbanEVDataset/UrbanEVDataset/20220901-20230228_station-raw/charge_5min")
csvlist = path2csv.glob("*.csv")


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

df = pd.DataFrame({'time','volume','busy','idle','fast_busy','slow_busy','s_price','e_price','duation'})
print(df)