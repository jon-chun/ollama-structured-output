import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ML/Clustering Libraries
from xgboost import XGBClassifier
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# Global Constants / File Paths
# -----------------------------------------------------------------------------

DATA_DIR = 'data'
INPUT_CLEAN_FILE = os.path.join(DATA_DIR, 'vignettes_renamed_clean.csv')
INPUT_VECTORIZED_FILE = os.path.join(DATA_DIR, 'vignettes_vectorized.csv')

CLUSTER_CT = 5  # number of difficulty clusters

# -----------------------------------------------------------------------------
# 1. Read Input Files
# -----------------------------------------------------------------------------
def read_input_files():
    """
    Attempts to read the cleaned and vectorized CSV files.
    Returns:
        df_clean (pd.DataFrame), df_vectorized (pd.DataFrame)
    """
    df_clean, df_vectorized = None, None

    print("[INFO] Attempting to read cleaned data file:", INPUT_CLEAN_FILE)
    try:
        df_clean = pd.read_csv(INPUT_CLEAN_FILE)
        print(f"[SUCCESS] Read {len(df_clean)} rows from {INPUT_CLEAN_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to read '{INPUT_CLEAN_FILE}': {e}")
        # Could either exit or continue if the data is not strictly required.
        # For now, let's exit so we don't proceed with partial data.
        sys.exit(1)

    print("[INFO] Attempting to read vectorized data file:", INPUT_VECTORIZED_FILE)
    try:
        df_vectorized = pd.read_csv(INPUT_VECTORIZED_FILE)
        print(f"[SUCCESS] Read {len(df_vectorized)} rows from {INPUT_VECTORIZED_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to read '{INPUT_VECTORIZED_FILE}': {e}")
        sys.exit(1)

    return df_clean, df_vectorized

# -----------------------------------------------------------------------------
# 2. Clustering Option A: XGBoost-Based Difficulty
# -----------------------------------------------------------------------------
def cluster_xgboost_difficulty(df_in: pd.DataFrame):
    """
    Clusters data into CLUSTER_CT buckets by difficulty, using XGBoost's predicted probabilities.
    Creates output files named 'vignette_cluster_diff-{i}.csv' (i=1..5).
    Assumes 'target' column exists in df_in.
    """
    print("\n[INFO] Starting Clustering Option A: XGBoost-based Difficulty")

    # Check if target exists
    if 'target' not in df_in.columns:
        print("[WARNING] 'target' column not found, skipping XGBoost difficulty clustering.")
        return

    # Separate features and target
    print("[DEBUG] Splitting features (X) and target (y).")
    y = df_in['target']
    X = df_in.drop(columns=['target'])

    print("[INFO] Training XGBoost Classifier...")
    xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
    xgb_clf.fit(X, y)

    # Difficulty measure: distance from 0.5 probability
    probs = xgb_clf.predict_proba(X)[:, 1]
    difficulty_scores = np.abs(probs - 0.5)

    # Attach difficulty scores
    df_in = df_in.copy()  # avoid modifying original
    df_in['difficulty_score'] = difficulty_scores

    # Sort rows by ascending difficulty
    df_sorted = df_in.sort_values(by='difficulty_score', ascending=True)

    # Basic binning into CLUSTER_CT buckets
    n = len(df_sorted)
    cluster_size = n // CLUSTER_CT
    clusters = []
    for i in range(n):
        c_id = i // cluster_size + 1
        c_id = min(c_id, CLUSTER_CT)  # handle last cluster edge case
        clusters.append(c_id)

    df_sorted['cluster_id'] = clusters

    # Save the clusters
    for clust_num in range(1, CLUSTER_CT + 1):
        subset = df_sorted[df_sorted['cluster_id'] == clust_num]
        out_file = f"vignette_cluster_diff-{clust_num}.csv"
        subset.to_csv(out_file, index=False)
        print(f"[INFO] Saved cluster {clust_num} with {len(subset)} rows to '{out_file}'")

# -----------------------------------------------------------------------------
# 3. Clustering Option B: Embeddings + KMeans
# -----------------------------------------------------------------------------
def cluster_embeddings_kmeans(df_in: pd.DataFrame):
    """
    Example: Use UMAP embeddings + KMeans to cluster the data into CLUSTER_CT clusters.
    Then label them 1..CLUSTER_CT (not strictly an 'easy'->'hard' measure unless we define it).
    """
    print("\n[INFO] Starting Clustering Option B: UMAP Embeddings + KMeans")

    # We'll use the entire df_in as numeric features (assuming it's from the vectorized dataset).
    # If 'target' is present, we'll drop it for pure clustering:
    df_cluster = df_in.copy()
    if 'target' in df_cluster.columns:
        df_cluster.drop(columns=['target'], inplace=True)

    # UMAP to reduce to 2D for simpler KMeans clustering
    print("[INFO] Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
    X_embedded = reducer.fit_transform(df_cluster)
    print("[DEBUG] Completed UMAP. Shape of X_embedded:", X_embedded.shape)

    # KMeans clustering
    print(f"[INFO] Clustering with KMeans into {CLUSTER_CT} clusters...")
    kmeans = KMeans(n_clusters=CLUSTER_CT, random_state=42)
    cluster_ids = kmeans.fit_predict(X_embedded)

    # Attach cluster labels
    df_cluster['cluster_id'] = cluster_ids + 1  # 1..CLUSTER_CT

    # Currently, we haven't defined a difficulty measure. If desired, you could:
    # - re-train a classifier on df_cluster
    # - compute local density
    # or simply treat cluster_id as a label for grouping.

    # Save each cluster
    for c_id in range(1, CLUSTER_CT + 1):
        subset = df_cluster[df_cluster['cluster_id'] == c_id]
        out_file = f"vignette_cluster_emb-{c_id}.csv"
        subset.to_csv(out_file, index=False)
        print(f"[INFO] Saved cluster {c_id} with {len(subset)} rows to '{out_file}'")

# -----------------------------------------------------------------------------
# 4. Clustering Option C: Agglomerative (Hierarchical) + Assign Difficulty
# -----------------------------------------------------------------------------
def cluster_agglomerative_with_difficulty(df_in: pd.DataFrame):
    """
    Performs hierarchical clustering (Ward's linkage) on df_in (assuming numeric columns).
    Then assigns difficulty levels (1=easy .. CLUSTER_CT=hard) using a logistic regression
    accuracy measure within each cluster, if 'target' is present.
    """
    print("\n[INFO] Starting Clustering Option C: Agglomerative Clustering + Difficulty Ranking")

    df_cluster = df_in.copy()
    # Separate target if available
    has_target = 'target' in df_cluster.columns
    if has_target:
        y = df_cluster['target']
        X = df_cluster.drop(columns=['target'])
    else:
        X = df_cluster

    # Perform Agglomerative Clustering
    print(f"[INFO] AgglomerativeClustering with n_clusters={CLUSTER_CT} ...")
    agg = AgglomerativeClustering(n_clusters=CLUSTER_CT, affinity='euclidean', linkage='ward')
    cluster_ids = agg.fit_predict(X)
    df_cluster['cluster_id'] = cluster_ids + 1  # 1..CLUSTER_CT

    # If we have a target, attempt to measure difficulty
    difficulty_dict = {}
    if has_target:
        print("[DEBUG] 'target' found. Calculating difficulty based on cluster-level accuracy.")
        for c_id in range(1, CLUSTER_CT + 1):
            subset = df_cluster[df_cluster['cluster_id'] == c_id]
            if len(subset) < 2:
                # Very small cluster, skip or treat as special
                difficulty_dict[c_id] = float('inf')  # artificially large difficulty
                continue

            X_sub = subset.drop(columns=['cluster_id', 'target'], errors='ignore')
            y_sub = subset['target']

            # Fit a quick logistic regression on this cluster
            clf = LogisticRegression()
            try:
                clf.fit(X_sub, y_sub)
                score = clf.score(X_sub, y_sub)  # cluster-level accuracy
                # define difficulty = 1 - accuracy
                difficulty = 1.0 - score
            except Exception as e:
                print(f"[WARNING] Could not fit LR in cluster {c_id}, defaulting difficulty=1.0: {e}")
                difficulty = 1.0

            difficulty_dict[c_id] = difficulty

        # Rank clusters by ascending difficulty
        cluster_ranking = sorted(difficulty_dict, key=difficulty_dict.get)

        # Remap cluster_id to difficulty level (1=easy..CLUSTER_CT=hard)
        final_mapping = {}
        for rank, cid in enumerate(cluster_ranking, start=1):
            final_mapping[cid] = rank

        df_cluster['diff_level'] = df_cluster['cluster_id'].map(final_mapping)

        # Save each difficulty level
        for diff_lvl in range(1, CLUSTER_CT + 1):
            subset = df_cluster[df_cluster['diff_level'] == diff_lvl]
            out_file = f"vignette_cluster_diff-{diff_lvl}.csv"
            subset.to_csv(out_file, index=False)
            print(f"[INFO] Saved difficulty level {diff_lvl} with {len(subset)} rows to '{out_file}'")
    else:
        print("[WARNING] 'target' not found. Assigning cluster_id without difficulty ranking.")
        # Just save each cluster as is
        for c_id in range(1, CLUSTER_CT + 1):
            subset = df_cluster[df_cluster['cluster_id'] == c_id]
            out_file = f"vignette_cluster_agglomerative-{c_id}.csv"
            subset.to_csv(out_file, index=False)
            print(f"[INFO] Saved cluster {c_id} with {len(subset)} rows to '{out_file}'")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    # Read Data
    df_clean, df_vectorized = read_input_files()

    print("\n[INFO] Data read complete. Beginning clustering procedures...")

    # -------------------------------------------------------------------------
    # A) XGBoost-based difficulty on the "clean" dataset (assuming it has 'target')
    # -------------------------------------------------------------------------
    cluster_xgboost_difficulty(df_clean)

    # -------------------------------------------------------------------------
    # B) Embedding + KMeans on the "vectorized" dataset
    # -------------------------------------------------------------------------
    cluster_embeddings_kmeans(df_vectorized)

    # -------------------------------------------------------------------------
    # C) Agglomerative Clustering on the "vectorized" dataset (to keep it numeric)
    # -------------------------------------------------------------------------
    cluster_agglomerative_with_difficulty(df_vectorized)

    print("\n[INFO] All clustering tasks completed.")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
