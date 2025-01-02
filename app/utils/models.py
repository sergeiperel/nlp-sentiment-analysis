from typing import List, Dict, Any

from pydantic import BaseModel


class ValDataset(BaseModel):
    X_val: List[List[float]]
    y_val: List[float]


class FitRequestItem(BaseModel):
    X_train: List[List[float]]
    y_train: List[float]
    hyperparameters: Dict[str, Any]
    val_dataset: ValDataset | None = None


class FitResponseItem(BaseModel):
    auc_roc: float
    tpr_train: List[float]
    fpr_train: List[float]
    auc_roc_val: float | None = None
    tpr_val: List[float] | None = None
    fpr_val: List[float] | None = None


class ModelListResponseItem(BaseModel):
    id: int
    model_type: str
    description: str


class PredictRequest(BaseModel):
    X: List[List[float]]


class PredictionResponseItem(BaseModel):
    predict: List[int]
    predict_prob: List[float]
