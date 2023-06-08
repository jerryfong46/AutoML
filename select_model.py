import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns


# you need to import or write more necessary modules or functions for e.g., cumulative gains chart, specificity, balanced accuracy etc.

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class ModelTuning:
    def __init__(self, feature_file, train_file, model_types):
        self.feature_file = feature_file
        self.train_file = train_file
        self.model_types = model_types
        self.load_data()  # load data to determine task
        self.task = (
            self.determine_task()
        )  # determine if task is classification or regression
        self.models = {
            model_type: self.get_model_instance(model_type)
            for model_type in model_types
        }
        self.feature_importances = {}
        self.grid = {
            "RandomForest": {
                "n_estimators": [100, 200, 500, 1000],
                "max_features": ["auto", "sqrt"],
                "max_depth": [4, 5, 6, 7, 8],
                "criterion": ["gini", "entropy"],
            },
            "XGBoost": {
                "n_estimators": [100, 200, 500, 1000],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
            },
            "LightGBM": {
                "n_estimators": [100, 200, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [5, 10, 15, -1],
                "num_leaves": [20, 30, 40, 50],
            },
            "Regression": {},  # No typical hyperparameters to tune for simple regression
            "Logistic": {
                "penalty": ["l1", "l2", "elasticnet", "none"],
                "C": np.logspace(-4, 4, 20),
                "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
            },
        }

    def determine_task(self):
        # If target variable has only 2 unique values, it is a classification task
        if self.data["TARGET"].nunique() == 2:
            return "classification"
        else:  # else, it is a regression task
            return "regression"

    def get_model_instance(self, model_type):
        if self.task == "classification":
            if model_type == "RandomForest":
                return RandomForestClassifier()
            elif model_type == "XGBoost":
                return XGBClassifier()
            elif model_type == "LightGBM":
                return LGBMClassifier()
            elif model_type == "Logistic":
                return LogisticRegression()
            elif model_type == "ZeroR":
                return DummyClassifier(strategy="most_frequent")
            else:
                raise ValueError("Invalid model type")
        else:  # regression models
            if model_type == "RandomForest":
                return RandomForestRegressor()
            elif model_type == "XGBoost":
                return XGBRegressor()
            elif model_type == "LightGBM":
                return LGBMRegressor()
            elif model_type == "Regression":
                return LinearRegression()
            elif model_type == "ZeroR":
                return DummyRegressor(strategy="mean")
            else:
                raise ValueError("Invalid model type")

    def load_data(self):
        self.features = pd.read_csv(self.feature_file)

        all_files = glob.glob(self.train_file)
        all_files.sort()  # ensure that the files are sorted

        data_frames = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df[self.features]  # select only the relevant features
            df["Period"] = os.path.basename(filename)[
                :-4
            ]  # get filename without extension as the 'Period'
            data_frames.append(df)

        self.data = pd.concat(data_frames, axis=0, ignore_index=True)

    def split_data(self):
        unique_periods = self.data["Period"].unique()

        if len(unique_periods) > 2:
            # For each unique period, we will train on all prior periods and validate on the current period
            self.splits = [
                (
                    np.where(self.data["Period"] < period)[0],
                    np.where(self.data["Period"] == period)[0],
                )
                for period in unique_periods
            ]
        else:
            # use StratifiedKFold for 2 periods or less
            skf = StratifiedKFold(n_splits=2)
            self.splits = list(
                skf.split(
                    self.data.drop(["TARGET", "Period"], axis=1), self.data["TARGET"]
                )
            )

    def tune_hyperparameters(self):
        for name, model in self.models.items():
            if name == "ZeroR":  # No hyperparameter tuning needed for ZeroR
                continue
            model_cv = RandomizedSearchCV(
                model, self.grid[name], cv=self.splits, scoring="roc_auc", n_jobs=-1
            )
            model_cv.fit(
                self.data.drop(["TARGET", "Period"], axis=1), self.data["TARGET"]
            )  # Exclude 'Period' column
            model.set_params(**model_cv.best_params_)
            self.feature_importances[name] = model.feature_importances_

    def model_eval(self):
        self.results = {}
        for name, model in self.models.items():
            model.fit(
                self.data.drop(["TARGET", "Period"], axis=1), self.data["TARGET"]
            )  # Exclude 'Period' column
            y_pred = model.predict(
                self.data.drop(["TARGET", "Period"], axis=1)
            )  # Exclude 'Period' column
            y_prob = model.predict_proba(self.data.drop(["TARGET", "Period"], axis=1))[
                :, 1
            ]  # Exclude 'Period' column
            self.results[name] = {
                "Parameters": model.get_params(),
                "Precision": precision_score(self.data["TARGET"], y_pred),
                "Recall": recall_score(self.data["TARGET"], y_pred),
                "F1": f1_score(self.data["TARGET"], y_pred),
                "AUC-ROC": roc_auc_score(self.data["TARGET"], y_prob),
                # Calculate the rest of the metrics here,
                "Confusion Matrix": confusion_matrix(self.data["TARGET"], y_pred),
            }

    def visualize_results(self):
        pass
        # Here, write your logic for AUC-ROC chart, Cumulative Gains Chart and Table, and Feature Importance Chart.

    def execute_pipeline(self):
        self.load_data()
        self.split_data()
        self.tune_hyperparameters()
        self.model_eval()
        self.visualize_results()


model_tuner = ModelTuning(
    feature_file="output/selected_features.csv", train_file="train_YYYYMM.csv"
)
model_tuner.execute_pipeline()
