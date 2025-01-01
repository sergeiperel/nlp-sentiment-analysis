from fastapi import FastAPI
import joblib
import pickle
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from transformers import AutoModelForTokenClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import time
import concurrent.futures
import signal
import os
import multiprocessing
import numpy as np
# Глобальные переменные для модели и токенизатора
model = None
tokenizer = None


# Словарь для хранения информации о моделях
models_info = {}
class SetActiveModelInput(BaseModel):
    model_id: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
class ModelInfo(BaseModel):
    name: str
    path: str
    status: str

class TrainingData(BaseModel):
    model_name: str
    features: list
    target: int
class TrainParams(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int

class TrainModelInput(BaseModel):
    model_name: str
    training_data: str

class PredictionInput(BaseModel):
    features: list

class FitModelInput(BaseModel):
    model_name: str
    model_type: str
    training_data: List[List[float]]  # Двумерный список для входных данных
    target_data: List[float]            # Массив целевых данных
    hyperparameters: dict                # Словарь гиперпараметров в формате JSON

class LoadModelInput(BaseModel):
    model_path: str
    model_name: str

class PredictInput(BaseModel):
    model_name: str
    features: List[float]

def load_model():
    global model
    with open("dt.pkl", "rb") as f:  # Замените на путь к вашей модели
        model = pickle.load(f)


#async def load_model():
#    global model, tokenizer
#    # Загрузка предобученной модели и токенизатора
#    model_name = "distilbert-base-uncased"
#    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)

#async def lifespan(app: FastAPI):
#    await load_model()
#    yield  # Это место, где приложение будет работать
#    global model, tokenizer
#    model = None
#    tokenizer = None
#lifespan=lifespan
app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/")
async def home():
    return {"message": "Сервер запущен, модель загружена."}


def train_model(X, y, model_name: str, model_type: str, model_params: dict):
    global model_info

    # Определяем и создаем модель
    if model_type == "LogisticRegression":
        model = LogisticRegression(**model_params)
    elif model_type == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(**model_params)
    elif model_type == "CatBoostClassifier":
        model = CatBoostClassifier(**model_params)
    else:
        raise ValueError("Unsupported model type")

    # Обучаем и сохраняем модель
    model.fit(X, y)
    joblib.dump(model, f"{model_name}.pkl")

    # Сохраняем информацию о модели
    model_info[model_name] = {
        "model_type": model_type,
        "hyperparameters": model_params
    }


@app.post("/fit/")
async def fit_model(input_data: FitModelInput):
    global training_process

    # Преобразование входных данных в массивы NumPy
    X = np.array(input_data.training_data)
    y = np.array(input_data.target_data)

    # Запустим процесс обучения
    training_process = multiprocessing.Process(
        target=train_model,
        args=(X, y, input_data.model_name, input_data.model_type, input_data.hyperparameters)
    )
    training_process.start()

    # Подождем 10 секунд
    training_process.join(timeout=10)

    # Если процесс все еще активен, убиваем его
    if training_process.is_alive():
        training_process.terminate()
        training_process.join()
        return {"message": "Обучение модели заняло слишком много времени и было прервано."}

    return {"message": f"Обучена модель {input_data.model_name}."}



class PredictionInput(BaseModel):
    features: list

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    data = pickle.loads(contents)

    prediction = model.predict(data)
    #model_name = input_data.model_name

    # Проверка, загружена ли модель
    #if model_name not in models_info:
    #   raise HTTPException(status_code=404, detail=f"Модель {model_name} не найдена.")

    #model = models_info[model_name]

    # Преобразование входных данных в массив NumPy
    #features_array = np.array(input_data.features).reshape(1, -1)

    # Получение предсказания
    #prediction = model.predict(features_array)

    return {"prediction": prediction.tolist()}


@app.get("/models/", response_model=Dict[str, ModelInfoResponse])
async def get_models():
    if not models_info:
        raise HTTPException(status_code=404, detail="Нет доступных моделей.")

    return {model_name: ModelInfoResponse(**info) for model_name, info in models_info.items()}
# Функция для загрузки модели и сохранения информации о ней
#def load_model(model_name: str, model_path: str):
#    try:
#        model = joblib.load(model_path)
#        models_info[model_name] = {
#            "path": model_path,
#            "status": "loaded"
#        }
#        return model
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))

# Пример загрузки моделей (замените на ваши пути)
#load_model("model1", "model1.pkl")
#load_model("model2", "model2.pkl")


@app.post("/set/")
async def set_active_model(input_data: SetActiveModelInput):
    global active_model_id
    if input_data.model_id in models_info:
        active_model_id = input_data.model_id
        return {"message": f"Active model set to {active_model_id}"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.post("/retrain/")
async def retrain(data: TrainingData):
    # Проверка на существование модели
    if data.model_name not in models_info:
        raise HTTPException(status_code=404, detail="Модель не найдена")
    # Проверка на корректность размера входных данных
    if len(data.features) != model.n_features_in_:
        raise HTTPException(status_code=400, detail="Неверное количество признаков")

    # Подготовка данных для дообучения
    X_new = np.array(data.features).reshape(1, -1)  # превращаем в 2D массив
    y_new = np.array([data.target])  # целевая метка

    # Дообучение модели
    model.fit(X_new, y_new)

    # Сохранение обновленной модели
    joblib.dump(model, 'model.pkl')

    return {"message": "Модель успешно дообучена"}