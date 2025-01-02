from fastapi import FastAPI
from typing import List
import joblib

app = FastAPI()

models_info = {
    "DecisionTreeClassifier": joblib.load('dt.pkl')
}
X = joblib.load('X.pkl')


@app.post('/predict/DecisionTree')
async def predict():

    prediction = models_info["DecisionTreeClassifier"].predict(X)
    return prediction


@app.get("/")
async def home():
    return {"message": "Сервер запущен, модель загружена."}





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
