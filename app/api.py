from fastapi import FastAPI
from typing import List
import joblib

app = FastAPI()


@app.post('/predict/DecisionTree')
async def predict(data: List[float]):

    model = joblib.load("models/DecisionTree.pkl")

    prediction = model.predict([data])[0]
    return prediction


@app.post('/predict/LogisticRegression')
async def predict(data: List[float]):

    model = joblib.load("models/LogReg.pkl")

    prediction = model.predict([data])[0]
    return prediction


@app.post('/predict/CatBoost')
async def predict(data: List[float]):

    model = joblib.load("models/CatBoost.pkl")

    prediction = model.predict([data])[0]
    return prediction
