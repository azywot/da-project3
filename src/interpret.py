import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap


def score_model(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    auc = round(roc_auc_score(y_true, y_pred), 4)
    return accuracy, f1, auc


def shap_tree_explainer(model: xgb.XGBClassifier, x: np.ndarray,
                        pred: np.ndarray) -> tuple:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
    shap.summary_plot(shap_values, x)
    return explainer, shap_values
