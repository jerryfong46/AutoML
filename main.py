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
        self.IMB_THRESHOLD = 0.95
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
        self.df = self.df.loc[:, self.df.var() != 0.0]
        return self.df

    def upper_case_columns(self):
        self.df.columns = map(str.upper, self.df.columns)
        return self.df

    def fill_target_nulls(self):
        self.df[self.target].fillna(0, inplace=True)
        return self.df

    def remove_ids(self):
        # Remove columns ending with '_ID' or '_NBR'
        self.df = self.df[self.df.columns[~self.df.columns.str.endswith(('_ID', '_NBR'))]]
        return self.df

    def normalize(self):
        self.df = pd.DataFrame(self.scaler.fit_transform(self.df), columns = self.df.columns)
        return self.df

    def denormalize(self, df_normalized):
        df_denormalized = pd.DataFrame(self.scaler.inverse_transform(df_normalized), columns=df_normalized.columns)
        return df_denormalized

    def balance_classes(self):
        # check if class imbalance exists
        counts = self.df[self.target].value_counts()
        if max(counts) / sum(counts) > self.IMB_THRESHOLD:
            majority_class = counts.idxmax()
            minority_class = counts.idxmin()
            df_majority = self.df[self.df[self.target] == majority_class]
            df_minority = self.df[self.df[self.target] == minority_class]
            
            # Downsample majority class until minority class is 5% of dataset
            downsample_ratio = int(len(df_minority) / (1-self.IMB_THRESHOLD))
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





    # Implement other preprocessing steps...


class Model:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.X = df.drop(target, axis=1)
        self.y = df[target]

    def fit_model(self):
        pass

    def feature_selection(self):
        pass

# Usage:

df = pd.read_csv(path + 'data.csv')

# Initialize your preprocessing class
target = 'TARGET'
preprocessor = DataPreprocessing(df, target)

# Apply preprocessing steps
df = preprocessor.label_encoding()
df = preprocessor.remove_high_nullity(threshold=0.8)
df = preprocessor.remove_zero_variance()
df = preprocessor.balance_classes()
df = preprocessor.limit_data_size()
df = preprocessor.feature_correlation()
df = preprocessor.normalize()

# Initialize your model class
model = Model(df, target)

# Fit models and perform feature selection
model.fit_model()
model.feature_selection()

# You can add more classes or methods as needed for your specific task
