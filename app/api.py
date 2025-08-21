import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/RandomForsetRegressorModel.joblib")

model, feature_order = joblib.load(MODEL_PATH)

app = FastAPI()

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_order]
    return df

@app.post("/predict_single/")
async def predict_single(features: dict):
    try:
        df = pd.DataFrame([features])
        df_processed = preprocess_input(df)
        prediction = model.predict(df_processed)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch/")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df_processed = preprocess_input(df)
        preds = model.predict(df_processed)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
