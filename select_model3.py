import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import h2o
import glob
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from h2o.automl import H2OAutoML
from h2o.backend import H2OLocalServer
from functools import wraps
import datetime
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, Image


def handle_unseen_categories(x, le):
    """Handle unseen categories in the test data by putting them into a new 'Other' category."""
    try:
        return le.transform([x])[0]
    except ValueError:
        return -1


class ModelTraining:
    def __init__(self, data_path, features_path, primary_key, target_key):
        self.features = pd.read_csv(features_path)
        self.data_path = data_path
        self.primary_key = primary_key.upper()
        self.target_key = target_key.upper()

        (
            self.data,
            self.validation,
            self.original_ratio,
            self.feature_mapping,
        ) = self._get_data(self.data_path)

        self.shap_data = self.data.copy()
        self.data = self._downsample_data(self.data)
        self._init_h2o()

    def _get_data(self, data_path):
        # Create a list of CSV files sorted by filename
        file_list = sorted(glob.glob(os.path.join(data_path, "*.csv")))

        # Ensure there's at least one CSV file in the directory
        if not file_list:
            raise ValueError(f"No CSV files found in {data_path}")

        # If there's only one file, split it into training and validation sets
        if len(file_list) == 1:
            full_data = pd.read_csv(file_list[0])
            data, validation = train_test_split(
                full_data,
                test_size=0.20,
                stratify=full_data[self.target_key],
                random_state=42,
            )
        else:
            # Concatenate all CSV files except the last one into a single DataFrame
            data = pd.concat(
                [pd.read_csv(file) for file in file_list[:-1]], ignore_index=True
            )
            # The last CSV file is considered as validation data
            validation = pd.read_csv(file_list[-1])

        # Extract feature names and their corresponding mapping
        feature_list = self.features.iloc[:, 0].tolist()
        feature_mapping = pd.Series(
            self.features.iloc[:, 1].values, index=self.features.iloc[:, 0]
        ).to_dict()

        # Append the target column to the feature list
        feature_list.append(self.target_key)

        # Convert column names to uppercase for consistency
        data.columns = data.columns.str.upper()
        validation.columns = validation.columns.str.upper()

        # Filter data and validation sets based on the feature list
        data = data[feature_list]
        validation = validation[feature_list]

        # Calculate the ratio of class distributions for the target column
        original_ratio = (
            data[self.target_key].value_counts()[0]
            / data[self.target_key].value_counts()[1]
        )

        # Return data, validation, original ratio, and the feature mapping
        return data, validation, original_ratio, feature_mapping

    def _downsample_data(
        self, data, majority_class=0, minority_class=1, majority_ratio=19
    ):
        """
        Downsample the majority class to achieve a specified ratio with the minority class.

        Parameters:
        - data (pd.DataFrame): The input data containing the target variable.
        - majority_class (int): The label of the majority class.
        - minority_class (int): The label of the minority class.
        - majority_ratio (int): The desired ratio of majority to minority class after downsampling.

        Returns:
        - pd.DataFrame: The downsampled data.
        """

        # Compute the count for each class in the target variable
        target_count = data[self.target_key].value_counts()

        # Ensure both classes exist in the data
        if minority_class not in target_count or majority_class not in target_count:
            raise ValueError(
                f"Data doesn't contain the expected classes {minority_class} and {majority_class}."
            )

        # Check if the minority class is below 1% of the data
        if target_count[minority_class] / target_count.sum() < 0.01:
            majority_data = data[data[self.target_key] == majority_class]
            minority_data = data[data[self.target_key] == minority_class]

            # Sample the majority class to achieve the desired ratio
            majority_downsampled = majority_data.sample(
                int(minority_data.shape[0] * majority_ratio)
            )

            # Concatenate the downsampled majority class with the minority class
            data = pd.concat([majority_downsampled, minority_data])

        return data

    def _init_h2o(self, max_mem_size="1G"):
        """
        Initialize the H2O environment and convert dataframes into H2O frames.

        Parameters:
        - max_mem_size (str): Maximum memory size for H2O initialization.

        Returns:
        None
        """
        # Initialize the H2O environment
        h2o.init(nthreads=-1, max_mem_size=max_mem_size)

        # Map pandas dtypes to H2O column types
        def get_h2o_type(dtype):
            if dtype == "object":
                return "enum"
            elif dtype == "float64":
                return "real"
            else:
                return "int"

        self.column_types = {
            col: get_h2o_type(self.data[col].dtype.name) for col in self.data.columns
        }

        # Ensure the validation set has the same types as the training set
        for col, col_type in self.column_types.items():
            if col_type == "enum":
                self.validation[col] = self.validation[col].astype("object")
            elif col_type == "real":
                self.validation[col] = self.validation[col].astype("float64")
            else:
                self.validation[col] = self.validation[col].astype("int")

        # Convert pandas dataframes to H2O frames
        self.data = h2o.H2OFrame(self.data, column_types=self.column_types)
        self.validation = h2o.H2OFrame(self.validation, column_types=self.column_types)

        # Convert target column to factor if it has 3 or fewer unique values
        if len(self.data[self.target_key].unique()) <= 3:
            self.data[self.target_key] = self.data[self.target_key].asfactor()
            self.validation[self.target_key] = self.validation[
                self.target_key
            ].asfactor()

    def _train_model(self, max_runtime_secs, final_train=False):
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            seed=42,
            exclude_algos=[
                "DeepLearning",
                "StackedEnsemble",
            ],  # Exclude for interpretability
        )

        if final_train:
            # concatenate training and validation datasets
            final_dataset = self.data.rbind(self.validation)
            aml.train(y=self.target_key, training_frame=final_dataset)
        else:
            aml.train(y=self.target_key, training_frame=self.data)

        return aml

    def _evaluate_model(self, aml):
        # Generate a timestamp for the current run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get model performance on the validation set
        perf = aml.leader.model_performance(self.validation)

        # You can choose a key metric, for example, AUC. Adapt accordingly if using another metric
        auc = round(perf.auc(), 4)

        # Create a directory name using the timestamp and AUC
        dir_name = f"model_{timestamp}_AUC_{auc}"

        # Create the directory
        output_dir = os.path.join("output", dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save the model to the directory
        model_path = h2o.save_model(model=aml.leader, path=output_dir, force=True)

        # Save performance metrics to the directory
        with open(os.path.join(output_dir, "val_performance.txt"), "w") as file:
            file.write(perf.__str__())

        return perf, model_path, output_dir

    # Additional specific plots can be added as needed (e.g., force plots, dependence plots)

    def _plot(self, perf, aml, output_dir):  # Accept output_dir as an argument
        # Roc curve
        roc_curve = perf.roc()
        auc_value = perf.auc()

        plt.figure()
        plt.plot(roc_curve[0], roc_curve[1])
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.text(0.6, 0.2, f"AUC = {auc_value:.2f}", fontsize=12)
        plt.savefig(os.path.join(output_dir, "val_auc_roc_plot.png"))
        plt.close()

        # Cumulative gains chart
        gains_lift = perf.gains_lift()
        plt.figure()
        plt.plot(gains_lift.as_data_frame()["cumulative_gain"])
        plt.title("Cumulative Gains Chart")
        plt.savefig(os.path.join(output_dir, "val_cumulative_gains_plot.png"))
        plt.close()

        # Variable importance
        importance = aml.leader.varimp(use_pandas=True)
        plt.figure()
        importance.plot(
            kind="bar",
            x="variable",
            y="relative_importance",
            legend=False,
            figsize=(14, 10),
        )
        plt.title("Variable Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_variable_importance_plot.png"))
        plt.close()

        # Lift Chart
        gains_lift_table = perf.gains_lift().as_data_frame()
        plt.figure(figsize=(10, 7))
        plt.plot(
            gains_lift_table["cumulative_data_fraction"],
            gains_lift_table["lift"],
            label="Lift",
            color="b",
        )
        plt.axhline(y=1, color="red", linestyle="--")
        plt.title("Lift Chart")
        plt.xlabel("Cumulative Data Fraction")
        plt.ylabel("Lift")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "val_lift_chart.png"))
        plt.close()

    def save_h2o_explain_to_pdf(self, aml, output_dir):
        validation_data = self.validation

        # Top variables comparison across top 10 models produced
        top_models = aml.leaderboard.sort("auc").head(10)
        fig, ax = plt.subplots(figsize=(12, 8))
        h2o.varimp_heatmap(top_models)
        output_path = os.path.join(output_dir, "var_imp_models.png")
        plt.savefig(output_path)
        plt.close(fig)

        # Get the leader model
        leader_model = aml.leader
        fig, ax = plt.subplots(figsize=(12, 8))
        leader_model.shap_summary_plot(validation_data)
        output_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.savefig(output_path)
        plt.close(fig)

        # Save pd plot for top variables
        varimps = leader_model.varimp()
        top_vars = [v[0] for v in varimps[:10]]
        for var in top_vars:
            # Generate the PD plot
            plt.figure(figsize=(10, 6))
            leader_model.pd_plot(validation_data, column=var)

            # Save the figure as a PNG
            png_path = os.path.join(output_dir, f"pd_plot_{var}.png")
            plt.savefig(png_path)
            plt.close()

    def train_and_shap(self, top_ids):
        data = self.data.as_data_frame()
        features = list(self.feature_mapping.keys())

        X = data.drop(self.target_key, axis=1)
        y = data[self.target_key]

        # Identify the categorical and numerical columns
        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(include=["number"]).columns

        # Fill missing values
        X[categorical_cols] = X[categorical_cols].fillna("Other")
        X[numerical_cols] = X[numerical_cols].fillna(0)
        y = y.fillna(0)

        # Identify the categorical columns
        # categorical_cols = X.columns[X.dtypes == 'object']

        # Apply label encoding for each categorical column
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # Train a random forest classifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # Load test data
        test_data = pd.read_csv(self.test_data_path)
        test_data.columns = test_data.columns.str.upper()
        test_data = test_data[features + [self.primary_key]]

        # Fill missing values in test data
        test_data[categorical_cols] = X[categorical_cols].fillna("Other")
        test_data[numerical_cols] = X[numerical_cols].fillna(0)

        test_data.fillna({"object": "Other", "number": 0}, inplace=True)
        test_data = test_data[test_data[self.primary_key].isin(top_ids)]

        # Apply label encoding to test data using label_encoders from training data
        for col, le in label_encoders.items():
            test_data[col] = test_data[col].apply(
                lambda x: handle_unseen_categories(x, le)
            )

        # Use SHAP to explain the model's predictions
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_data)[1]

        # Get top 3 features for each instance in the test set
        top_features = np.argsort(np.abs(shap_values), axis=1)[:, -3:]

        # Create a DataFrame with the ID, and the top 3 features and their values for each instance
        output = pd.DataFrame(
            columns=[self.primary_key, "Variable 1", "Variable 2", "Variable 3"]
        )

        for i in range(top_features.shape[0]):
            output.loc[i, self.primary_key] = test_data.iloc[i][self.primary_key]
            for j in range(3):
                feature_name = X.columns[top_features[i, j]]
                feature_description = self.feature_mapping.get(
                    feature_name, feature_name
                )  # Use the feature name as a fallback
                output.loc[
                    i, f"Variable {j+1}"
                ] = f"{feature_description}: {test_data.iloc[i][feature_name]}"

        return output

    def run(self):
        # train model with train dataset
        aml = self._train_model()
        perf, model_path = self._evaluate_model(aml)
        self._plot(perf, aml)

        # train final model with train+validation dataset
        aml_final = self._train_model(final_train=True)
        final_model_path = h2o.save_model(
            model=aml_final.leader, path="output", force=True
        )

        best_model = h2o.load_model(final_model_path)
        result = self.predict_with_best_model(best_model)
        result = result.reset_index(drop=True)
        shap_output = self.train_and_shap(
            top_ids=list(result.head(3000)[self.primary_key])
        )
        final_result = pd.merge(shap_output, result, on=self.primary_key, how="inner")
        return final_result

    def get_predictor_vars(self):
        """Return necessary variables for prediction."""
        return {
            "feature_mapping": self.feature_mapping,
            "column_types": self.column_types,
            "primary_key": self.primary_key,
        }


class ModelPredictor:
    def __init__(self, feature_mapping, column_types, primary_key, test_data_path):
        self.feature_mapping = feature_mapping
        self.column_types = column_types
        self.primary_key = primary_key
        self.test_data_path = test_data_path

    def predict_with_best_model(self, best_model):
        features = list(self.feature_mapping.keys())
        df_test = pd.read_csv(self.test_data_path)
        df_test.columns = map(str.upper, df_test.columns)
        best_model = h2o.load_model(best_model)

        missing_columns = [col for col in features if col not in df_test.columns]
        for col in missing_columns:
            print(f"{col} column is missing. Included with null values.")
            df_test[col] = None

        df_test = df_test[features + [self.primary_key]]

        test_column_types = {
            col: model_predictor.column_types[col]
            for col in df_test.columns
            if col in self.column_types
        }

        df_test = h2o.H2OFrame(df_test, column_types=test_column_types)

        ids = df_test[self.primary_key].as_data_frame()
        predictions = best_model.predict(df_test)

        scores = predictions["p1"].as_data_frame()
        result = pd.concat([ids, scores], axis=1)
        result = result.sort_values(by="p1", ascending=False)
        result = result.rename(columns={"p1": "SCORE"})

        return result

    def get_most_recent_model_path(self, directory):
        # List all files in the directory
        files = [os.path.join(directory, file) for file in os.listdir(directory)]

        # Filter out non-model directories
        model_files = [file for file in files if "automl" in file.lower()]

        # Sort the files based on their modification times
        model_files.sort(key=os.path.getmtime, reverse=True)

        # Return the path of the most recent model
        return model_files[0] if model_files else None


if __name__ == "__main__":
    # Set up model trainer
    model_trainer = ModelTraining(
        features_path="output/selected_features.csv",
        data_path="input_data",
        primary_key="ID",
        target_key="TARGET",
    )

    # Train and validate model
    aml = model_trainer._train_model(max_runtime_secs=60)
    perf, model_path, output_dir = model_trainer._evaluate_model(aml)
    model_trainer.save_h2o_explain_to_pdf(aml, output_dir)

    print(f"Model saved to {model_path}")
    model_trainer._plot(perf, aml, output_dir)

    # Train final model with all datasets
    aml_final = model_trainer._train_model(max_runtime_secs=60, final_train=True)
    os.makedirs("final_model", exist_ok=True)
    final_model_path = h2o.save_model(
        model=aml_final.leader, path="final_model", force=True
    )

    # Load final model and make predictions
    predictor_vars = model_trainer.get_predictor_vars()
    model_predictor = ModelPredictor(
        **predictor_vars, test_data_path="data/test.csv"
    )  # Path to test data

    # Use model to make predictions
    best_model = model_predictor.get_most_recent_model_path(
        "final_model"
    )  # Uses most recent model saved into directory
    predictions = model_predictor.predict_with_best_model(best_model)
    print(predictions)
