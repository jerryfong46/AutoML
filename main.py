# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler


path = 'data/'

# Load data

from sklearn.utils import resample
from scipy.stats import pointbiserialr, chi2_contingency
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

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

        return self.df

    def clean(self):
        self.df.columns = map(str.upper, self.df.columns)
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

    def feature_correlation(self):
        correlations_num = self.calc_pointbiserialr()
        correlations_cat = self.calc_chi2_contingency()
        # Merge the two dictionaries
        correlations = {**correlations_num, **correlations_cat}
        # Keep only top 250 features
        sorted_corr = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)[:250]
        selected_cols = [x[0] for x in sorted_corr]
        
        # Ensure the target column is included
        if self.target not in selected_cols:
            selected_cols.append(self.target)

        self.df = self.df.loc[:, selected_cols]
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

        return self.df


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

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

        rfe_rf = RFE(estimator=RandomForestClassifier(), n_features_to_select=200, step=1)
        rfe_rf.fit(self.X, self.y)
        rfe_rf_cols = self.X.columns[rfe_rf.support_]
        rfe_rf_importances = rfe_rf.estimator_.feature_importances_

        rfe_lr = RFE(estimator=LogisticRegression(max_iter=5000), n_features_to_select=200, step=1)
        rfe_lr.fit(self.X, self.y)
        rfe_lr_cols = self.X.columns[rfe_lr.support_]
        rfe_lr_importances = np.abs(rfe_lr.estimator_.coef_)[0]

        model = xgb.XGBClassifier()
        model.fit(self.X, self.y)
        xgb_cols = self.X.columns
        xgb_importances = model.feature_importances_

        rf = RandomForestClassifier().fit(self.X, self.y)
        permutation_result = permutation_importance(rf, self.X, self.y, n_repeats=10, random_state=0)
        permutation_cols = self.X.columns
        permutation_importances = permutation_result.importances_mean

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

        # get common features
        common_features = list(set.intersection(*[set(method) for method in top_features]))

        # if common features < 50, get from rfe_rf_cols and xgb_cols
        if len(common_features) < 50:
            alternates = list(zip(rfe_rf_cols, xgb_cols))
            for alternate in alternates:
                for feature in alternate:
                    if feature not in common_features:
                        cor_matrix = self.X[common_features + [feature]].corr().abs()
                        if all(cor_matrix[feature][:-1] < 0.7):
                            common_features.append(feature)
                            if len(common_features) == 50:
                                break
                if len(common_features) == 50:
                    break
        return common_features



df = pd.read_csv(path + 'data.csv')

# Initialize your preprocessing class
target = 'TARGET'
preprocessor = DataPreprocessing(df, target)

# Apply preprocessing steps
df = preprocessor.clean()
df = preprocessor.remove_high_nullity(threshold=0.8)
df = preprocessor.remove_zero_variance()
df = preprocessor.balance_classes(threshold=0.95)
df = preprocessor.limit_data_size()
df = preprocessor.label_encoding()
df = preprocessor.feature_correlation()
df = preprocessor.variable_clustering()
df = preprocessor.normalize()

model = FeatureSelector(df.drop(columns=[target]), df[target])
final_vars = model.process()
