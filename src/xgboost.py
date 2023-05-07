import xgboost as xgb


def create_xgb_model(criteria_nr: int) -> xgb.XGBClassifier:
    xgb_params = {
        "max_depth": criteria_nr * 2,
        "eta": 0.1,
        "nthread": 2,
        "seed": 0,
        "eval_metric": "rmse",
        "monotone_constraints": "(" + ",".join(["1"] * criteria_nr) + ")",
        "n_estimators": 1,
    }
    model = xgb.XGBClassifier(**xgb_params)
    return model
