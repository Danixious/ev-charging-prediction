import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import sqlite3

conn = sqlite3.connect("D:/EV_prediction/ev-charging-prediction/notebooks/ev.db")
df = pd.read_sql("SELECT * FROM ev_data", conn)
conn.close()

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

# #linerRegression
# model = LinearRegression()
# model = model.fit(X_train,y_train)
# y_pred = model.predict(x_test)

# mse = mean_squared_error(y_test,y_pred)
# mae = mean_absolute_error(y_test,y_pred)
# r2_score = r2_score(y_test,y_pred)

# print(mse)
# print(mae)
# print(r2_score)
# joblib.dump(model,'D:/EV_prediction/ev-charging-prediction/models/linearRegressionModel.joblib')

#RandomForest
rf_model = RandomForestRegressor()
rf_model = rf_model.fit(X_train,y_train)
rf_y_pred = rf_model.predict(x_test)

mse = mean_squared_error(y_test,rf_y_pred)
mae = mean_absolute_error(y_test,rf_y_pred)
r2_score = r2_score(y_test,rf_y_pred)
CV = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')

print(f"mse: {mse}")
print(f"mae: {mae}")
print(f"r2_score: {r2_score}")
print(f"CV: {CV}")

feature_order = X.columns.tolist()
joblib.dump((rf_model,X_train.columns.tolist()),'D:/EV_prediction/ev-charging-prediction/models/RandomForsetRegressorModel.joblib')

# #XGBoost
# xgb_model = XGBRegressor(n_estimators =100,learning_rate = 0.1, random_state = 42)
# xgb_model = xgb_model.fit(X_train,y_train)
# xgb_pred = xgb_model.predict(x_test)

# mse = mean_squared_error(y_test, xgb_pred)
# mae = mean_absolute_error(y_test, xgb_pred)
# r2 = r2_score(y_test, xgb_pred)
# cv = cross_val_score(xgb_model, X, y, cv=5)

# print(f"MSE: {mse}")
# print(f"MAE: {mae}")
# print(f"R2 Score: {r2}")
# print(f"Cross-Validation Scores: {cv}")
# print(f"Mean CV Score: {cv.mean()}")

# joblib.dump(xgb_model, "D:/EV_prediction/ev-charging-prediction/models/XGBoostRegressorModel.joblib")