print('Executing code...')

# Import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.inspection import permutation_importance

from scipy.stats import pointbiserialr, chi2_contingency, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

import xgboost as xgb

from collections import defaultdict
import itertools

import matplotlib.backends.backend_pdf

class DataPreprocessing:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.scaler = StandardScaler()  # Create a StandardScaler instance

    def label_encoding(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        return self.df

    def remove_high_nullity(self, threshold):
        self.df = self.df[self.df.columns[self.df.isnull().mean() < threshold]]
        print(f'Removing vars with high nullity - {len(self.df.columns)} variables remaining.')
        return self.df

    def remove_zero_variance(self):
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Select non-numeric columns
        non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns

        # Calculate variance for numeric columns and identify zero variance columns
        numeric_zero_var_cols = [col for col in numeric_cols if self.df[col].var() == 0.0]
        # Identify zero variance non-numeric columns
        non_numeric_zero_var_cols = [col for col in non_numeric_cols if self.df[col].nunique() <= 1]

        # Concatenate lists
        zero_var_cols = numeric_zero_var_cols + non_numeric_zero_var_cols

        # Drop zero variance columns
        self.df = self.df.drop(columns=zero_var_cols)

        print(f'Removing vars with zero variance - {len(self.df.columns)} variables remaining.')

        return self.df

    def clean(self):
        self.df.columns = self.df.columns.str.upper()  # Vectorized string operation
        self.df = self.df.loc[:, ~self.df.columns.str.endswith(('_ID', '_NBR'))].copy()
        self.df.fillna(0, inplace=True)
        return self.df


    def normalize(self):
        self.scalers = {}  # Initialize the dictionary of scalers
        for col in self.df.columns:
            if (
                (self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64') and 
                len(self.df[col].unique()) >= 10 and 
                not any(substr in col for substr in ["CNT", "IND", "CHG", "PCT", "3M", "6M"])
            ):
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler  # Store the scaler used for this column
        return self.df

    def denormalize(self, df_normalized):
        for col in df_normalized.columns:
            if col in self.scalers:  # Only denormalize columns that were originally normalized
                df_normalized[col] = self.scalers[col].inverse_transform(df_normalized[[col]])
        return df_normalized

    def balance_classes(self, threshold):
        # check if class imbalance exists
        counts = self.df[self.target].value_counts()
        if max(counts) / sum(counts) > threshold:
            majority_class = counts.idxmax()
            minority_class = counts.idxmin()
            df_majority = self.df[self.df[self.target] == majority_class]
            df_minority = self.df[self.df[self.target] == minority_class]
            
            # Downsample majority class until minority class is 5% of dataset
            downsample_ratio = int(len(df_minority) / (1-threshold))
            df_majority_downsampled = resample(df_majority, replace=False, n_samples=downsample_ratio, random_state=42)
            
            self.df = pd.concat([df_majority_downsampled, df_minority])
        return self.df

    def limit_data_size(self):
        if len(self.df) > 100000:
            self.df = resample(self.df, replace=False, n_samples=100000, random_state=42)
        return self.df

    def calc_pointbiserialr(self):
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        num_cols = num_cols.drop(self.target)  # remove target from the feature list
        correlations = {col: pointbiserialr(self.df[col].values, self.df[self.target].values)[0] for col in num_cols}
        return correlations

    def calc_chi2_contingency(self):
        cat_cols = self.df.select_dtypes(include=['object', 'bool', 'category']).columns
        chi2_vals = {}
        for col in cat_cols:
            contingency_table = pd.crosstab(self.df[col], self.df[self.target])
            chi2_vals[col] = chi2_contingency(contingency_table)[0]
        return chi2_vals

    def feature_correlation(self, keep_num):
        correlations_num = self.calc_pointbiserialr()
        correlations_cat = self.calc_chi2_contingency()
        # Merge the two dictionaries
        correlations = {**correlations_num, **correlations_cat}

        # Keep only top X features
        sorted_corr = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)[:keep_num]
        selected_cols = [x[0] for x in sorted_corr]
        
        # Ensure the target column is included
        if self.target not in selected_cols:
            selected_cols.append(self.target)

        self.df = self.df.loc[:, selected_cols]

        print(f'Creating correlation matrix - {len(self.df.columns)} variables selected.')
        return self.df

    def variable_clustering(self):
        # Exclude the target variable from the clustering
        df_cluster = self.df.drop(columns=[self.target])
        
        # Compute the correlation matrix
        corr_matrix = df_cluster.corr().abs()
        # Turn the correlation matrix into a distance matrix
        dist_matrix = 1 - corr_matrix

        # Perform hierarchical/agglomerative clustering
        clusters = linkage(squareform(dist_matrix), method='average')

        # Form flat clusters from the hierarchical clustering defined by the linkage matrix
        num_clusters = df_cluster.shape[1] // 3
        cluster_labels = fcluster(clusters, num_clusters, criterion='maxclust')

        # Select the most representative variable from each cluster
        selected_features = []
        for i in range(1, num_clusters + 1):
            cluster_vars = [var for var, cluster in zip(df_cluster.columns, cluster_labels) if cluster == i]
            # Select the variable with the highest sum of correlations with other variables in the cluster
            var_correlations = corr_matrix.loc[cluster_vars, cluster_vars].sum()
            selected_features.append(var_correlations.idxmax())

        # Update the DataFrame to include only the selected features, along with the target variable
        self.df = self.df[selected_features + [self.target]]

        print(f'Performing variable clustering - {len(self.df.columns)} variables selected.')
        return self.df
    
    def remaining_cols(self):
        return f'{len(self.df.columns)} variables selected.'

class FeatureSelector:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def rfe_rf(self):
        selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=200, step=1)
        selector = selector.fit(self.X, self.y)
        return self.X.columns[selector.support_]

    def rfe_lr(self):
        selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=200, step=1)
        selector = selector.fit(self.X, self.y)
        return self.X.columns[selector.support_]

    def xgboost(self):
        model = xgb.XGBClassifier(use_label_encoder=False)
        model.fit(self.X, self.y)
        importance = model.feature_importances_
        idx = np.argsort(importance)[-200:]
        return self.X.columns[idx]

    def permutation_importance(self):
        rf = RandomForestClassifier().fit(self.X, self.y)
        result = permutation_importance(rf, self.X, self.y, n_repeats=10, random_state=0)
        sorted_idx = result.importances_mean.argsort()[-200:]
        return self.X.columns[sorted_idx]

    def process(self):

        final_vars = []

        def get_top_features(cols, importances):
            # Sort features by importance
            sorted_idx = np.argsort(importances)[::-1]
            sorted_cols = [cols[i] for i in sorted_idx]
            to_drop = []
            for i, col in enumerate(sorted_cols):
                # Compare only with more important features
                cor_matrix = self.X[sorted_cols[:i+1]].corr().abs()
                # If current feature is highly correlated with any preceding feature
                if any(cor_matrix[col][:-1] > 0.7):
                    to_drop.append(col)
            sorted_cols = [col for col in sorted_cols if col not in to_drop]
            return sorted_cols[:3]  # Return top 3 features

        print('Selecting features using RFE with Random Forest...')
        rfe_rf = RFE(estimator=RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=100), n_features_to_select=200, step=1)
        rfe_rf.fit(self.X, self.y)
        rfe_rf_cols = self.X.columns[rfe_rf.support_]
        rfe_rf_importances = rfe_rf.estimator_.feature_importances_
        final_vars.extend(get_top_features(rfe_rf_cols.to_list(), rfe_rf_importances))

        print('Selecting features using RFE with Logistic Regression...')
        rfe_lr = RFE(estimator=LogisticRegression(max_iter=5000, n_jobs=-1), n_features_to_select=200, step=1)
        rfe_lr.fit(self.X, self.y)
        rfe_lr_cols = self.X.columns[rfe_lr.support_]
        rfe_lr_importances = np.abs(rfe_lr.estimator_.coef_)[0]
        final_vars.extend(get_top_features(rfe_lr_cols.to_list(), rfe_lr_importances))

        print('Selecting features using XGBoost...')
        model = xgb.XGBClassifier(n_jobs=-1)
        model.fit(self.X, self.y)
        xgb_cols = self.X.columns
        xgb_importances = model.feature_importances_
        final_vars.extend(get_top_features(xgb_cols.to_list(), xgb_importances))

        print('Selecting features using permutation importance...')
        rf = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=25).fit(self.X, self.y)
        permutation_result = permutation_importance(rf, self.X, self.y, n_repeats=10, random_state=0)
        permutation_cols = self.X.columns
        permutation_importances = permutation_result.importances_mean
        final_vars.extend(get_top_features(permutation_cols.to_list(), permutation_importances))

        final_vars = list(set(final_vars))  # Remove duplicates

        methods = [(rfe_rf_cols.to_list(), rfe_rf_importances), 
                (rfe_lr_cols.to_list(), rfe_lr_importances), 
                (xgb_cols.to_list(), xgb_importances), 
                (permutation_cols.to_list(), permutation_importances)]

        top_features = []
        for cols, importances in methods:
            # Sort features by importance
            sorted_idx = np.argsort(importances)[::-1]
            sorted_cols = [cols[i] for i in sorted_idx]
            to_drop = []
            for i, col in enumerate(sorted_cols):
                # Compare only with more important features
                cor_matrix = self.X[sorted_cols[:i+1]].corr().abs()
                # If current feature is highly correlated with any preceding feature
                if any(cor_matrix[col][:-1] > 0.7):
                    to_drop.append(col)
            sorted_cols = [col for col in sorted_cols if col not in to_drop]

            # get top 50 from each method
            top_features.append(sorted_cols[:50])

        print('Getting common features from all methods...')

        # get common features, add to final_vars
        common_features = list(set.intersection(*[set(method) for method in top_features]))
        final_vars.extend(common_features)
        final_vars = list(set(final_vars))  # Remove duplicates

        # if common features < 50, get from rfe_rf_cols and xgb_cols
        if len(final_vars) < 50:
            alternates = list(zip(rfe_rf_cols, xgb_cols))
            alternates = list(itertools.chain(*alternates))  # Flatten the list
            for feature in alternates:
                if feature not in final_vars:
                    cor_matrix = self.X[final_vars + [feature]].corr().abs()
                    if all(cor_matrix[feature][:-1] < 0.7):
                        final_vars.append(feature)
                        if len(final_vars) == 50:
                            break
        
        final_vars = list(set(final_vars))  # Remove duplicates after adding features
        return(final_vars)




class Model:
    def __init__(self, model_type, X, y):
        self.model_type = model_type
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train(self):
        print('Assessing final features...')
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=42)
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(n_jobs=-1, eval_metric='logloss')
        else:
            raise Exception("Invalid model_type; available types: 'random_forest', 'xgboost'")
        
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate(self):
        print(classification_report(self.y_test, self.y_pred))
        print(confusion_matrix(self.y_test, self.y_pred))

    def feature_importance(self, top_n):
        if self.model_type in ['random_forest', 'xgboost']:
            feature_importances = self.model.feature_importances_
            feature_importances = pd.Series(feature_importances, index=self.X.columns)
            top_features = feature_importances.sort_values(ascending=False)[:top_n]
            return top_features
        else:
            raise Exception(f"No feature importances available for model type: {self.model_type}")

    def plot_importance(self, top_n=30, pdf_pages=None):
        importances = self.feature_importance(top_n)
        
        fig = plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        importances.sort_values().plot(kind='barh', colormap='Blues_r')
        plt.xlabel('Relative Importance')
        plt.tight_layout()

        # If pdf_pages is not None, save the figure to the pdf
        if pdf_pages is not None:
            pdf_pages.savefig(fig)
        else:
            plt.show()

    def plot_features(self, n_bins, pdf_pages=None):
        data = self.X.copy()
        data['TARGET'] = self.y

        for feature in self.X.columns:
            fig = plt.figure(figsize=(10, 6))
            unique_vals = data[feature].nunique()
            if unique_vals <= n_bins:
                # Group by each unique value if less than or equal to 10
                feature_avg_target = data.groupby(feature)['TARGET'].mean().round(2)
                bars = sns.barplot(x=feature_avg_target.index, y=feature_avg_target.values, palette="Blues_r")
            else:
                # If more than 10 unique values, bin into 10 buckets of equal size
                data[feature + '_bins'] = pd.qcut(data[feature], q=n_bins, duplicates='drop')
                # Transform bin ranges from interval to string format
                data[feature + '_bins'] = data[feature + '_bins'].apply(lambda x: f'{x.left} to {x.right}')
                binned_avg_target = data.groupby(feature + '_bins')['TARGET'].mean().round(2)
                bars = sns.barplot(x=binned_avg_target.index, y=binned_avg_target.values, palette="Blues_r")

            # Add value labels on top of each bar
            for rect in bars.patches:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2, height, f'{height}', ha='center', va='bottom')

            plt.title(f'Average TARGET per {feature}')
            plt.xlabel(feature)
            plt.ylabel('Average TARGET')
            plt.xticks(rotation=45)  # Rotate x-axis labels
            plt.tight_layout()
            
            # If pdf_pages is not None, save the figure to the pdf
            if pdf_pages is not None:
                pdf_pages.savefig(fig)
                plt.close(fig)  # Close the figure after saving
            else:
                plt.show()
            
    def export_to_pdf(self, filename, n_bins=10):
        with matplotlib.backends.backend_pdf.PdfPages(filename) as pdf:
            # Plot and save feature importance
            self.plot_importance(n_bins, pdf_pages=pdf)
            # Plot and save feature charts
            self.plot_features(n_bins=n_bins, pdf_pages=pdf)


def read_and_union_csvs(directory="."):
    # Get all files in directory
    all_files = os.listdir(directory)
    # Filter for csv files
    csv_files = [file for file in all_files if file.endswith('.csv')]
    
    # Initialize an empty dataframe to store all data
    combined_df = pd.DataFrame()
    
    # Iterate over all csv files
    for file in csv_files:
        # Read the csv file
        df = pd.read_csv(file)
        # Append the data to the combined dataframe
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        
    return combined_df

if __name__ == '__main__':

    # Read and union all csv files in the current directory
    df = read_and_union_csvs()

    while True:
        print('Enter the name of the target column (case sensitive), or "escape" to stop:')
        target = input()

        if target.lower() == 'escape':
            print('User stopped the process.')
            break
        elif target in df.columns:
            preprocessor = DataPreprocessing(df, target)
            break
        else:
            print(f'{target} is not a column in the DataFrame. Please try again.')

    # Apply preprocessing steps
    df = preprocessor.clean()
    df = preprocessor.remove_high_nullity(threshold=0.8)
    df = preprocessor.remove_zero_variance()
    df = preprocessor.balance_classes(threshold=0.95)
    df = preprocessor.limit_data_size()
    df = preprocessor.label_encoding()
    df = preprocessor.remove_zero_variance()
    df = preprocessor.feature_correlation(keep_num=250)
    df = preprocessor.variable_clustering()
    df = preprocessor.normalize()

    # Define your data and target variable
    X = df.drop(columns=[target])
    y = df[target]

    model = FeatureSelector(X, y)
    final_vars = model.process()

    # Denormalize the data and select final features
    df = preprocessor.denormalize(df)
    X = df[final_vars]

    # Create an instance of the Model class and train it
    model_rf = Model('random_forest', X, y)
    model_rf.train()

    # Extract top features to pdf
    model_rf.export_to_pdf('random_forest_plots.pdf')


