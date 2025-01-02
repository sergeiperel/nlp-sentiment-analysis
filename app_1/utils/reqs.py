import os
from typing import Any

import numpy as np
import requests
from scipy.sparse import coo_matrix

HOST = os.environ.get('FASTAPI_HOST', "http://127.0.0.1:8001/")

def train_request(x_train: coo_matrix,
                  y_train: np.ndarray,
                  x_val: coo_matrix,
                  y_val: np.ndarray,
                  hyperparams: dict[str, Any])->tuple:
    url = f'{HOST}/fit'
    payload = {
        "X_train": x_train.todense().tolist(),
        "y_train": y_train.tolist(),
        "hyperparameters": hyperparams,
    }
    if x_val is not None and y_val is not None:
        payload['val_dataset'] = {}
        payload['val_dataset']["X_val"] = x_val.todense().tolist()
        payload['val_dataset']["y_val"] = y_val.tolist()
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    json_resp = resp.json()
    auc_roc = json_resp.get("auc_roc")
    tpr_train = np.asarray(json_resp.get('tpr_train'))
    fpr_train = np.asarray(json_resp.get('fpr_train'))
    if json_resp.get("auc_roc_val") is not None:
        auc_roc_val = json_resp.get("auc_roc_val")
        tpr_val = np.asarray(json_resp.get('tpr_val'))
        fpr_val = np.asarray(json_resp.get('fpr_val'))
    else:
        auc_roc_val, tpr_val, fpr_val = None, None, None
    return auc_roc, tpr_train, fpr_train, auc_roc_val, tpr_val, fpr_val

def list_request()->list[dict[str, str]]:
    url = f'{HOST}/models'
    resp = requests.get(url)
    resp.raise_for_status()
    json_resp = resp.json()
    return json_resp

def set_model(id_: int)->bool:
    url = f'{HOST}/set'
    resp = requests.post(f'{url}/{id_}')
    if resp.ok:
        return True
    return False


def inference_request(x_test: coo_matrix)->tuple:
    url = f'{HOST}/predict'
    payload = {
        "X": x_test.todense().tolist()
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    json_resp = resp.json()
    y_pred = np.asarray(json_resp.get('predict'))
    y_prob = np.asarray(json_resp.get('predict_prob'))
    return y_pred, y_prob



    



