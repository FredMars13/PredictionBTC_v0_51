# Version: V0.51
import numpy as np
import pandas as pd

def _to_proba_dict(class_names, probs_row):
    return {cls: float(p) for cls, p in zip(class_names, probs_row)}

def predict_trend(model, x_now_row: pd.DataFrame, class_names):
    probs = model.predict_proba(x_now_row)
    probs_row = probs[0]
    idx = int(np.argmax(probs_row))
    label = class_names[idx]
    return label, _to_proba_dict(class_names, probs_row)
