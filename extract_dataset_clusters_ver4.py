import os
import sys
import time
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
# 1. Read & Prepare Input Files
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
        print(f"[SUCCESS] Read {len(df_clean)} rows from '{INPUT_CLEAN_FILE}'")
    except Exception as e:
        print(f"[ERROR] Failed to read '{INPUT_CLEAN_FILE}': {e}")
        sys.exit(1)

    print("[INFO] Attempting to read vectorized data file:", INPUT_VECTORIZED_FILE)
    try:
        df_vectorized = pd.read_csv(INPUT_VECTORIZED_FILE)
        print(f"[SUCCESS] Read {len(df_vectorized)} rows from '{INPUT_VECTORIZED_FILE}'")
    except Exception as e:
        print(f"[ERROR] Failed to read '{INPUT_VECTORIZED_FILE}': {e}")
        sys.exit(1)

    return df_clean, df_vectorized

def convert_object_columns_to_category(df_in: pd.DataFrame, drop_text_cols=True):
    """
    Convert feasible object-type columns to category dtype so that
    XGBoost can handle them with 'enable_categorical=True'.
    Optionally drop text-heavy columns such as 'short_text_summary' and 'long_text_summary'.
    
    Returns a new DataFrame ready for XGBoost training or other clustering.
    """
    df_out = df_in.copy()

    # Potentially drop text-heavy columns
    if drop_text_cols:
        text_cols = ["short_text_summary", "long_text_summary"]
        for col in text_cols:
            if col in df_out.columns:
                print(f"[DEBUG] Dropping text column '{col}' for XGBoost classification or clustering.")
                df_out.drop(columns=[col], inplace=True)

    print("[INFO] Converting object columns to 'category' dtype (if feasible)...")
    # Identify object columns
    obj_cols = df_out.select_dtypes(['object']).columns

    for col in obj_cols:
        unique_vals = df_out[col].nunique(dropna=True)
        if unique_vals < 200:  # arbitrary threshold
            print(f"[DEBUG] Converting '{col}' to category (unique vals={unique_vals}).")
            df_out[col] = df_out[col].astype('category')
        else:
            print(f"[WARNING] Column '{col}' has {unique_vals} unique values, which may be large.")
            print("[WARNING] Consider label encoding or dropping if it's too big for your methods.")

    return df_out

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
    start_time = time.time()

    # Check if target exists
    if 'target' not in df_in.columns:
        print("[WARNING] 'target' column not found, skipping XGBoost difficulty clustering.")
        return

    # Separate features and target
    print("[DEBUG] Splitting features (X) and target (y).")
    y = df_in['target']
    X = df_in.drop(columns=['target'])

    # Convert object columns -> category, drop large text columns
    print("[INFO] Preparing DataFrame for XGBoost (converting objects to categories, dropping text).")
    X_prepped = convert_object_columns_to_category(X, drop_text_cols=True)

    print("[INFO] Training XGBoost Classifier with enable_categorical=True ...")
    try:
        xgb_clf = XGBClassifier(
            n_estimators=100,
            random_state=42,
            enable_categorical=True,  # requires XGBoost >= 1.7
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_clf.fit(X_prepped, y)
    except ValueError as ve:
        print("[ERROR] XGBoost training failed due to data type issues:", ve)
        return

    # Difficulty measure: distance from 0.5 probability
    print("[INFO] Computing difficulty scores...")
    probs = xgb_clf.predict_proba(X_prepped)[:, 1]
    difficulty_scores = np.abs(probs - 0.5)

    # Merge difficulty scores into original DataFrame
    df_clustered = df_in.copy()
    df_clustered['difficulty_score'] = difficulty_scores

    # Sort rows by ascending difficulty
    df_sorted = df_clustered.sort_values(by='difficulty_score', ascending=True)

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

    elapsed = time.time() - start_time
    print(f"[INFO] XGBoost-based clustering completed in {elapsed:.2f} seconds.")

# -----------------------------------------------------------------------------
# 3. Clustering Option B: Embeddings + KMeans
# -----------------------------------------------------------------------------
def cluster_embeddings_kmeans(df_in: pd.DataFrame):
    """
    Example: Use UMAP embeddings + KMeans to cluster the data into CLUSTER_CT clusters.
    Then label them 1..CLUSTER_CT.
    Note: This is purely unsupervised; there's no 'difficulty' measure unless you define one.
    """
    print("\n[INFO] Starting Clustering Option B: UMAP Embeddings + KMeans")
    start_time = time.time()

    df_cluster = df_in.copy()
    if 'target' in df_cluster.columns:
        print("[DEBUG] Dropping 'target' for unsupervised clustering.")
        df_cluster.drop(columns=['target'], inplace=True)

    # Make sure data is numeric. If object columns exist, we either convert or drop them
    df_cluster = convert_object_columns_to_category(df_cluster, drop_text_cols=True)

    # For UMAP, we typically feed numeric data. Convert category -> codes:
    cat_cols = df_cluster.select_dtypes(['category']).columns
    for c in cat_cols:
        df_cluster[c] = df_cluster[c].cat.codes

    print("[INFO] Performing UMAP dimensionality reduction (n_components=2)... This might take a few seconds.")
    reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
    X_embedded = reducer.fit_transform(df_cluster)
    print(f"[DEBUG] UMAP completed. Shape: {X_embedded.shape}")

    # KMeans clustering
    print(f"[INFO] Clustering with KMeans into {CLUSTER_CT} clusters...")
    kmeans = KMeans(n_clusters=CLUSTER_CT, random_state=42)
    cluster_ids = kmeans.fit_predict(X_embedded)

    # Attach cluster labels to the original DataFrame
    df_out = df_in.copy()
    df_out['cluster_id'] = cluster_ids + 1  # 1..CLUSTER_CT

    # Save each cluster
    for c_id in range(1, CLUSTER_CT + 1):
        subset = df_out[df_out['cluster_id'] == c_id]
        out_file = f"vignette_cluster_emb-{c_id}.csv"
        subset.to_csv(out_file, index=False)
        print(f"[INFO] Saved cluster {c_id} with {len(subset)} rows to '{out_file}'")

    elapsed = time.time() - start_time
    print(f"[INFO] Embeddings + KMeans clustering completed in {elapsed:.2f} seconds.")

# -----------------------------------------------------------------------------
# 4. Clustering Option C: Agglomerative (Hierarchical) + Assign Difficulty
# -----------------------------------------------------------------------------
def cluster_agglomerative_with_difficulty(df_in: pd.DataFrame):
    """
    Performs hierarchical clustering (Ward's linkage) on df_in (assuming numeric or category).
    Then assigns difficulty levels (1=easy..CLUSTER_CT=hard) using a logistic regression
    accuracy measure within each cluster, if 'target' is present.

    NOTE: In newer scikit-learn versions, the parameter is `metric` instead of `affinity`.
    """
    print("\n[INFO] Starting Clustering Option C: Agglomerative Clustering + Difficulty Ranking")
    start_time = time.time()

    df_cluster = df_in.copy()
    has_target = 'target' in df_cluster.columns
    if has_target:
        y = df_cluster['target']
        X = df_cluster.drop(columns=['target'])
    else:
        X = df_cluster

    print("[INFO] Preparing data (object->category, dropping text) for AgglomerativeClustering.")
    X_prepped = convert_object_columns_to_category(X, drop_text_cols=True)

    # Convert any categories -> numeric codes
    cat_cols = X_prepped.select_dtypes(['category']).columns
    for c in cat_cols:
        X_prepped[c] = X_prepped[c].cat.codes

    print("[INFO] Performing AgglomerativeClustering (Ward linkage, n_clusters=5). This may take time for large data.")
    # NOTE: In scikit-learn >= 1.4, use `metric='euclidean'` instead of `affinity='euclidean'`.
    agg = AgglomerativeClustering(n_clusters=CLUSTER_CT, metric='euclidean', linkage='ward')
    cluster_ids = agg.fit_predict(X_prepped)
    
    df_out = df_cluster.copy()
    df_out['cluster_id'] = cluster_ids + 1  # 1..CLUSTER_CT

    if has_target:
        print("[DEBUG] 'target' found. Calculating difficulty based on cluster-level logistic regression accuracy.")
        difficulty_dict = {}
        for c_id in range(1, CLUSTER_CT + 1):
            subset = df_out[df_out['cluster_id'] == c_id]
            if len(subset) < 2:
                # Very small cluster, skip or treat as special
                print(f"[WARNING] Cluster {c_id} has <2 rows, forcing high difficulty.")
                difficulty_dict[c_id] = float('inf')
                continue

            X_sub = subset.drop(columns=['cluster_id', 'target'], errors='ignore')
            y_sub = subset['target']

            # Convert any categories again if present
            for col in X_sub.select_dtypes(['category']).columns:
                X_sub[col] = X_sub[col].cat.codes

            # Fit logistic regression
            clf = LogisticRegression()
            try:
                clf.fit(X_sub, y_sub)
                score = clf.score(X_sub, y_sub)  # cluster-level accuracy
                # define difficulty = 1 - accuracy
                difficulty = 1.0 - score
            except Exception as e:
                print(f"[WARNING] Could not fit LR for cluster {c_id}, default difficulty=1.0. Error: {e}")
                difficulty = 1.0

            difficulty_dict[c_id] = difficulty

        # Rank clusters by ascending difficulty
        cluster_ranking = sorted(difficulty_dict, key=difficulty_dict.get)
        final_mapping = {}
        for rank_idx, cid in enumerate(cluster_ranking, start=1):
            final_mapping[cid] = rank_idx

        df_out['diff_level'] = df_out['cluster_id'].map(final_mapping)

        # Save each difficulty level
        for diff_lvl in range(1, CLUSTER_CT + 1):
            subset = df_out[df_out['diff_level'] == diff_lvl]
            out_file = f"vignette_cluster_diff-{diff_lvl}.csv"
            subset.to_csv(out_file, index=False)
            print(f"[INFO] Saved difficulty level {diff_lvl} with {len(subset)} rows to '{out_file}'")
    else:
        print("[WARNING] 'target' not found. Assigning cluster_id without difficulty ranking.")
        for c_id in range(1, CLUSTER_CT + 1):
            subset = df_out[df_out['cluster_id'] == c_id]
            out_file = f"vignette_cluster_agglomerative-{c_id}.csv"
            subset.to_csv(out_file, index=False)
            print(f"[INFO] Saved cluster {c_id} with {len(subset)} rows to '{out_file}'")

    elapsed = time.time() - start_time
    print(f"[INFO] Agglomerative-based clustering completed in {elapsed:.2f} seconds.")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    print("[INFO] Starting cluster extraction script. Please wait...\n")

    # Read Data
    df_clean, df_vectorized = read_input_files()
    print("[INFO] Data read complete. Beginning clustering procedures...")

    # Option A: XGBoost-based difficulty on the "clean" dataset
    cluster_xgboost_difficulty(df_clean)

    # Option B: Embedding + KMeans on the "vectorized" dataset
    cluster_embeddings_kmeans(df_vectorized)

    # Option C: Agglomerative Clustering on the "vectorized" dataset
    cluster_agglomerative_with_difficulty(df_vectorized)

    print("\n[INFO] All clustering tasks completed. Exiting now.")

if __name__ == "__main__":
    main()
