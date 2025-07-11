import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib

df = pd.read_csv("D:/EV_prediction/ev-charging-prediction/data/Cleaned_data.csv")

encoder = OneHotEncoder(sparse_output=False)
one_hot_encoder = encoder.fit_transform(df[['weekday']])
one_hot_df = pd.DataFrame(one_hot_encoder,columns = encoder.get_feature_names_out(['weekday']))
df_encoded = pd.concat([df.drop(['weekday'],axis = 1), one_hot_df],axis = 1)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded),columns = df_encoded.columns)
print(df_scaled.head())

y = df_scaled['volume']
X = df_scaled.drop('volume', axis = 1)

X_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

model = LinearRegression()
model = model.fit(X_train,y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2_score = r2_score(y_test,y_pred)

print(mse)
print(mae)
print(r2_score)
joblib.dump(model,'D:/EV_prediction/ev-charging-prediction/models/linearRegressionModel.joblib')
