import os
import pandas as pd
import numpy as np
import pathlib as Path
from xgboost import XGBClassifier

INPUT_CLEAN_FILE = os.path.join('data', 'vignettes_renamed_clean.csv')
INPUT_VECTORIZED_FILE = os.path.join('data', 'vignettes_vectorized.csv')

CLUSTER_CT = 5

# Example: reading the vectorized file
df_clean = pd.read_csv(INPUT_CLEAN_FILE)
df_vectorized = pd.read_csv(INPUT_VECTORIZED_FILE)

# OPTIONAL: If needed, define X as the feature matrix, 
# and y if you still have the target for reference. 
# (Assuming 'target' might be in the dataset.)
if 'target' in df_clean.columns:
    y = df_clean['target']
    X = df_clean.drop(columns=['target'])
else:
    X = df_clean.copy()

# If you only want to cluster on certain columns,
# filter X as needed:
# e.g. X = X[['col1','col2',...]]




# Traain XGBoost Classifier
xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
xgb_clf.fit(X, y)  # requires y (the binary target)

# Get Difficulty Scores from Predicted Probabilities
probs = xgb_clf.predict_proba(X)[:, 1]  # Probability of the positive class
difficulty_scores = np.abs(probs - 0.5)

# Cluster Rows into n-Buckets by Difficulty
df_clean['difficulty_score'] = difficulty_scores
df_sorted = df_clean.sort_values(by='difficulty_score', ascending=True)

# number of rows per cluster
n = len(df_sorted)
cluster_size = n // CLUSTER_CT

# label each row with cluster_id from 1 (least difficult) to 5 (most difficult)
clusters = []
for i in range(n):
    # integer division by cluster_size
    c_id = i // cluster_size + 1
    # boundary condition: last cluster might get extra rows
    c_id = min(c_id, 5)
    clusters.append(c_id)

df_sorted['cluster_id'] = clusters

# Save Clusters
for clust_num in range(1, 6):
    subset = df_sorted[df_sorted['cluster_id'] == clust_num]
    subset.to_csv(f"vignette_cluster_diff-{clust_num}.csv", index=False)
