import numpy as np
import xgboost as xgb
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
from src.nb_UTA import Uta
from src.helpers import append_output, Hook


def add_measures(measures: dict, results: tuple, model_name: str, mode: str) -> dict:
    if model_name not in measures:
        measures[model_name] = dict()
    measures[model_name][mode] = {
        "accuracy": round(results[0], 4),
        "f1": round(results[1], 4),
        "auc": round(results[2], 4),
    }
    return measures


def score_model(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    auc = round(roc_auc_score(y_true, y_pred), 4)
    return accuracy, f1, auc


def shap_tree_explainer(
    model: xgb.XGBClassifier, x: np.ndarray, pred: np.ndarray
) -> tuple:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    # np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
    shap.summary_plot(shap_values, x, color="coolwarm")
    return explainer, shap_values


def getSimpleInput(val: float, criteria_nr: int):
    return torch.FloatTensor([[val] * criteria_nr]).view(1, 1, -1).cpu()


def get_marginal_values(
    model: Uta, criteria_nr: int
) -> tuple[list[torch.FloatTensor, np.ndarray]]:
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
    fig, axs = plt.subplots(1, criteria_nr, figsize=(25, 5))

    for i in range(criteria_nr):
        axs[i].plot(xs, outs[:, i], color="deeppink")
        axs[i].set_ylabel("marginal value $u_{0}(a_i)$".format(i + 1), fontsize=14)
        axs[i].set_xlabel("performance $g_{0}(a_i)$".format(i + 1), fontsize=14)
        axs[i].set_title("Criterion #{}".format(i + 1), fontsize=16)

    plt.suptitle("Marginal values for each attribute (ai = Criterion #i)", fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_measures(data):
    metrics = ["accuracy", "f1", "auc"]
    models = data.keys()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    width = 0.5

    for i, metric in enumerate(metrics):
        train_data = [data[model]["train"][metric] for model in models]
        test_data = [data[model]["test"][metric] for model in models]

        x = range(len(models))
        axs[i].bar(
            [xi - (width / 2) for xi in x],
            train_data,
            width=width,
            color="b",
            alpha=0.5,
            label="train",
        )
        axs[i].bar(
            [xi + (width / 2) for xi in x],
            test_data,
            width=width,
            color="deeppink",
            alpha=0.5,
            label="test",
        )

        axs[i].set_xlabel("Model")
        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(models)
        axs[i].set_title(f"{metric.capitalize()} (Train/Test)")
        axs[i].legend()

    plt.show()
