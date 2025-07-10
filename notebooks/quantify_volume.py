import pandas as pd
import numpy as np

df = pd.read_csv("D:/EV_prediction/ev-charging-prediction/data/Cleaned_data.csv")
bins = 3
labels = [f'Bin {i+1}' for i in range(bins)]
df['volume_bin'] = pd.qcut(df['volume'],q=bins,labels=labels,precision= 0)
output_path = "D:/EV_prediction/ev-charging-prediction/data/Quantified_data.csv"
df.to_csv(output_path,index = False)

print("quantification complete")