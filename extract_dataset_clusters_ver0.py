import os
import pandas as pd
import numpy as np
import pathlib as Path
from xgboost import XGBClassifier

INPUT_CLEAN_FILE = os.path.join('data', 'vignettes_renamed_clean.csv')
INPUT_VECTORIZED_FILE = os.path.join('data', 'vignettes_vectorized.csv')

CLUSTER_CT = 5

# PREPARE: Cleaned Dataset
# =====+++++

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


# CLUSTER OPTION A: by XGBoost Feature Importance / Explainability via Probabilities
# ==========

# Traain XGBoost Classifier
xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
xgb_clf.fit(X, y)  # requires y (the binary target)

# Get Difficulty Scores from Predicted Probabilities
probs = xgb_clf.predict_proba(X)[:, 1]  # Probability of the positive class
difficulty_scores = np.abs(probs - 0.5)

# Cluster Rows into CLUSTER_CT buckets by Difficulty
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



# CLUSTER OPTION B: by Embedding Similarity
# ==========

import umap

# Generate or Load Embeddings
reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
X_embedded = reducer.fit_transform(X)
# X_embedded is now Nx2 or NxD (if you choose more components)


# Compute Difficulty via Distance
# TODO

# Cluster intn CLUSTER_CT Buckets
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_ids = kmeans.fit_predict(X_embedded)
df['cluster_id'] = cluster_ids + 1  # to make them 1..5 # TODO: Clarify df_clean or df_vectorized or other df

# Assigne Cluster difficulty scores 1-5
# TODO

# Save Clusters
for clust_num in range(1, 6):
    subset = df[df['cluster_id'] == clust_num] # TODO: Clarify df_clean or df_vectorized or other df
    subset.to_csv(f"vignette_cluster_diff-{clust_num}.csv", index=False)


# CLUSTER OPTION C: by Agglomerative (Hierarchical) Clustering
# ==========

# Setup
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster_ids = agg.fit_predict(X)  # or X_embedded if you want to reduce dimensionality first
df['cluster_id'] = cluster_ids + 1  # TODO: Clarify df_clean or df_vectorized or other df


# Assign Difficulty Levels
# Hypothetical approach if we have y (labels):
from sklearn.linear_model import LogisticRegression
import numpy as np

difficulty_dict = {}
for c_id in range(1, 6):
    subset = df[df['cluster_id'] == c_id]  # TODO: Clarify df_clean or df_vectorized or other df
    X_sub = subset.drop(columns=['cluster_id', 'target'], errors='ignore')
    y_sub = subset['target'] if 'target' in subset.columns else None

    if y_sub is not None:
        clf = LogisticRegression()
        clf.fit(X_sub, y_sub)
        score = clf.score(X_sub, y_sub)  # "accuracy" or other metric
        # Define "difficulty" as 1 - score (lower accuracy => higher difficulty)
        difficulty_dict[c_id] = 1.0 - score
    else:
        # Purely unsupervised scenario: fallback to cluster sizes or
        # compute some measure of intra-cluster variance, etc.
        difficulty_dict[c_id] = X_sub.var().sum()  # example metric

# Rank cluster_ids by ascending difficulty
cluster_ranking = sorted(difficulty_dict, key=difficulty_dict.get)

# Re-map cluster_id to 1..5, where 1 = easiest, 5 = hardest
final_mapping = {}
for rank, c_id in enumerate(cluster_ranking, start=1):
    final_mapping[c_id] = rank

df['diff_level'] = df['cluster_id'].map(final_mapping)  # TODO: Clarify df_clean or df_vectorized or other df


# Save Each Cluster
for diff_lvl in range(1, 6):
    subset = df[df['diff_level'] == diff_lvl]  # TODO: Clarify df_clean or df_vectorized or other df
    subset.to_csv(f"vignette_cluster_diff-{diff_lvl}.csv", index=False)


