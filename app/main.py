import json
import logging
import os
from logging.handlers import RotatingFileHandler
from multiprocessing import Queue, Process
from typing import List, Any

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from starlette.responses import JSONResponse

from utils.models import (
    FitRequestItem,
    ModelListResponseItem,
    FitResponseItem,
    PredictRequest,
    PredictionResponseItem,
)

model_path = "models"

TRAIN_TIMEOUT = 300

app = FastAPI()


def configure_logging():
    logging.basicConfig(level=logging.INFO)
    handler = RotatingFileHandler("logs/app.log", maxBytes=2000, backupCount=10)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger("backend")
    logger.addHandler(handler)
    return logger


logger = configure_logging()

loaded_models = []
active_model = {}

def create_model(name: str, hyper: dict[str, Any])->BaseEstimator:
    if name == 'DT':
        model = DecisionTreeClassifier(**hyper)
    elif name == 'LR':
        model = LogisticRegression(**hyper)
    elif name == 'KNN':
        model = KNeighborsClassifier(**hyper)
    else:
        model = None
    return model


def train_model(
    x_train,
    y_train,
    model: BaseEstimator,
    file_path: str,
    q: Queue,
    x_val=None,
    y_val=None,
):
    try:
        model.fit(x_train, y_train)
    except (ValueError, MemoryError, TypeError) as e:
        q.put(e)
        return
    y_pred = model.predict_proba(x_train)[:, 1]
    auc_roc = roc_auc_score(y_train, y_pred)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    result = FitResponseItem(
        auc_roc=auc_roc, fpr_train=fpr.tolist(), tpr_train=tpr.tolist()
    )
    if x_val is not None and y_val is not None:
        y_pred = model.predict_proba(x_val)[:, 1]
        auc_roc = roc_auc_score(y_val, y_pred)
        fpr, tpr, thresholds = roc_curve(y_val, y_pred)
        result.auc_roc_val = auc_roc
        result.fpr_val = fpr.tolist()
        result.tpr_val = tpr.tolist()
    q.put(result)
    try:
        joblib.dump(model, file_path)
        return JSONResponse({'message': 'Model was saved successful!'})
    except (FileNotFoundError, PermissionError, ValueError) as e:
        q.put(e)


@app.on_event("startup")
def load_model():
    with open(os.path.join(model_path, "models.json")) as f:
        loaded_models.extend(json.load(f))
    for item in loaded_models:
        model = joblib.load(os.path.join(model_path, item["file_name"]))
        item["model"] = model


@app.post("/fit", status_code=201, response_model=FitResponseItem)
def fit_model(request: FitRequestItem):
    if active_model is None:
        logger.error("Active model wasn't set.")
        raise HTTPException(status_code=422, detail="Please set active model.")
    hyperparams = request.hyperparameters
    X_train = np.asarray(request.X_train)
    y_train = np.asarray(request.y_train)
    if request.val_dataset:
        X_val = np.asarray(request.val_dataset.X_val)
        y_val = np.asarray(request.val_dataset.y_val)
    else:
        X_val = None
        y_val = None
    model = create_model(active_model['type'], hyperparams)
    q = Queue()
    file_path = os.path.join(model_path, active_model["file_name"])
    p = Process(
        target=train_model, args=(X_train, y_train, model, file_path, q, X_val, y_val)
    )
    p.start()
    p.join(timeout=TRAIN_TIMEOUT)
    if p.is_alive():
        p.terminate()
        logger.error("Training took too long and was terminated")
        raise HTTPException(
            status_code=422, detail="Training took too long and was terminated"
        )
    result = q.get()
    if isinstance(result, Exception):
        logger.error(str(result))
        raise HTTPException(detail=str(result), status_code=422)
    active_model["model"] = joblib.load(file_path)
    logger.info(f"Model ID: {active_model['type']} was trained.")
    return result



@app.post("/set/{model_id}", response_model=ModelListResponseItem, status_code=200)
def set_model(model_id: int):
    for model in loaded_models:
        if model["id"] == model_id:
            global active_model
            active_model = model
            logger.info(f"Model ID: {active_model['id']} is set!")
            return ModelListResponseItem(id=model['id'],
                                         model_type=model['type'],
                                         description=model["description"])
    logger.error(f"Model ID: {model_id} was not found.")
    raise HTTPException(detail="Model not found", status_code=422)


@app.get("/models", response_model=List[ModelListResponseItem], status_code=200)
def list_models():
    model_list = [
        ModelListResponseItem(id=model["id"],
                              description=model["description"],
                              model_type=model["type"])
        for model in loaded_models
    ]
    return model_list


@app.post("/predict", response_model=PredictionResponseItem, status_code=200)
def predict(request: PredictRequest):
    if not active_model:
        logging.error("Active model wasn't set.")
        HTTPException(status_code=422, detail="Please set active model.")
    model = active_model["model"]
    try:
        predictions = model.predict(request.X)
        predictions_prob = model.predict_proba(request.X)[:, 1]
    except ValueError as e:
        logging.error(str(e))
        raise HTTPException(detail=str(e), status_code=422)
    return PredictionResponseItem(
        predict=predictions.tolist(), predict_prob=predictions_prob.tolist()
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
