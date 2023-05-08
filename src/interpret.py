import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
from src.nb_UTA import Uta
from src.helpers import append_output, Hook


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
    shap.summary_plot(shap_values, x, color='coolwarm')
    return explainer, shap_values

# NOTE: not used
def shap_deep_explainer(model: nn.Module, x: np.ndarray) -> tuple:
    x_tensor = torch.from_numpy(x)
    explainer = shap.DeepExplainer(model, x_tensor)
    shap_values = explainer.shap_values(x_tensor)
    shap.summary_plot(shap_values, x_tensor, color='coolwarm')
    return explainer, shap_values

def getSimpleInput(val: float, criteria_nr: int):
    return torch.FloatTensor([[val] * criteria_nr]).view(1, 1, -1).cpu()


def get_marginal_values(model: Uta, criteria_nr: int) -> tuple[list[torch.FloatTensor, np.ndarray]]:
    hook = Hook(model.method.criterionLayerCombine, append_output)
    xs = []
    with torch.no_grad():
        for i in range(21):
            val = i / 20.0
            x = getSimpleInput(val, criteria_nr)
            xs.append(val)
            model(x)

    outs = np.array(torch.stack(hook.stats)[:, 0].detach().cpu())
    outs = outs * model.method.sum_layer.weight.detach().numpy()[0]
    outs = outs[::3] - outs[::3][0]
    outs = outs / outs[-1].sum()
    return xs, outs

def plot_marginal_values_ann_utadis(model: Uta, criteria_nr: int):
    xs, outs = get_marginal_values(model, criteria_nr)
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(criteria_nr):
        axs[i].plot(xs, outs[:, i], color="deeppink")
        axs[i].set_ylabel("marginal value $u_{0}(a_i)$".format(i + 1), fontsize=14)
        axs[i].set_xlabel("performance $g_{0}(a_i)$".format(i + 1), fontsize=14)

    plt.suptitle("Marginal values for each attribute (ai = Criterion #i)", fontsize=20)
    plt.tight_layout()
    plt.show()