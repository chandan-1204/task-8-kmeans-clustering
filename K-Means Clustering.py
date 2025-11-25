#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# ---------- CONFIG ----------
RAW_PATH = "C:\\Users\\chand\\Downloads\\Mall_Customers.csv"  # path to uploaded dataset (keeps from conversation)
OUT_DIR = 'kmeans_outputs'
RANDOM_STATE = 42
MAX_K = 10    # evaluate k from 1..MAX_K (silhouette requires >=2)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- LOAD ----------
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Update RAW_PATH or place file there.")

# Try to read CSV or Excel
if RAW_PATH.lower().endswith(('.xls', '.xlsx')):
    df = pd.read_excel(RAW_PATH)
else:
    df = pd.read_csv(RAW_PATH)

print("Loaded:", RAW_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------- CLEAN ----------
# drop columns that are all-NaN or Unnamed index columns
df = df.dropna(axis=1, how='all')
df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
print("After dropping empty/unnamed cols, shape:", df.shape)

# Prefer numeric columns for KMeans (distance-based). If you want to include categorical, encode them separately.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found. Convert or encode categorical features before clustering.")

X = df[numeric_cols].copy()
print("Numeric features used:", numeric_cols)

# Fill missing numeric values with median
imp = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Save preprocessor objects for reuse
joblib.dump(imp, os.path.join(OUT_DIR, 'imputer.joblib'))
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.joblib'))

# ---------- PCA for 2D visualization (if >2 features) ----------
n_features = X_scaled.shape[1]
use_pca = n_features > 2
if use_pca:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_vis = pca.fit_transform(X_scaled)
    joblib.dump(pca, os.path.join(OUT_DIR, 'pca_2comp.joblib'))
    print("PCA: explained variance ratios:", pca.explained_variance_ratio_)
else:
    X_vis = X_scaled  # already 2D or 1D (1D will not plot clusters meaningfully)

# Save cleaned numeric sample
pd.DataFrame(X_imputed, columns=X.columns).head(500).to_csv(os.path.join(OUT_DIR, 'cleaned_numeric_sample.csv'), index=False)

# ---------- Elbow method (inertia) ----------
inertias = []
ks = list(range(1, MAX_K + 1))
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"k={k} inertia={km.inertia_:.3f}")

plt.figure(figsize=(8,5))
plt.plot(ks, inertias, '-o')
plt.xlabel('k (number of clusters)')
plt.ylabel('Inertia (sum of squared distances)')
plt.title('Elbow Method: Inertia vs k')
plt.xticks(ks)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'elbow_inertia.png'))
plt.close()
print("Saved elbow plot to", os.path.join(OUT_DIR, 'elbow_inertia.png'))

# ---------- Silhouette scores (k >= 2) ----------
sil_scores = []
ks_sil = list(range(2, MAX_K + 1))
for k in ks_sil:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    print(f"k={k} silhouette={sil:.4f}")

plt.figure(figsize=(8,5))
plt.plot(ks_sil, sil_scores, '-o', color='purple')
plt.xlabel('k (number of clusters)')
plt.ylabel('Silhouette score')
plt.title('Silhouette Score vs k')
plt.xticks(ks_sil)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'silhouette_scores.png'))
plt.close()
print("Saved silhouette plot to", os.path.join(OUT_DIR, 'silhouette_scores.png'))

# ---------- Choose k automatically by max silhouette (fallback to elbow) ----------
if len(sil_scores) > 0:
    best_k = ks_sil[int(np.argmax(sil_scores))]
    print("Best k by silhouette:", best_k)
else:
    # if only k=1 was possible, fallback to 2
    best_k = 2
    print("No silhouette scores computed; defaulting best_k =", best_k)

# ---------- Fit final KMeans ----------
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
labels = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_
print("Final KMeans fitted. Cluster centers (scaled space) shape:", centers.shape)

# Save model
joblib.dump(kmeans, os.path.join(OUT_DIR, f'kmeans_k{best_k}.joblib'))

# ---------- Save cluster assignments to CSV ----------
out_df = df.copy().reset_index(drop=True)
out_df['cluster'] = labels
out_df.to_csv(os.path.join(OUT_DIR, f'data_with_clusters_k{best_k}.csv'), index=False)
print("Saved clustered data to", os.path.join(OUT_DIR, f'data_with_clusters_k{best_k}.csv'))

# ---------- Visualize clusters in 2D ----------
plt.figure(figsize=(8,6))
palette = sns.color_palette('tab10', n_colors=best_k)
if use_pca:
    # transform cluster centers to PCA space
    centers_pca = pca.transform(centers)
    for i in range(best_k):
        idx = labels == i
        plt.scatter(X_vis[idx,0], X_vis[idx,1], s=40, alpha=0.6, label=f'cluster {i}', color=palette[i])
    plt.scatter(centers_pca[:,0], centers_pca[:,1], s=200, marker='X', c='black', label='centers')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title(f'KMeans clusters (k={best_k}) - PCA(2)')
else:
    for i in range(best_k):
        idx = labels == i
        plt.scatter(X_vis[idx,0], X_vis[idx,1], s=40, alpha=0.6, label=f'cluster {i}', color=palette[i])
    plt.title(f'KMeans clusters (k={best_k})')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'clusters_k{best_k}.png'), dpi=150)
plt.close()
print("Saved cluster visualization to", os.path.join(OUT_DIR, f'clusters_k{best_k}.png'))

# ---------- Silhouette by sample (optional plot for best_k) ----------
try:
    sample_silhouette = silhouette_score(X_scaled, labels)
    print(f"Silhouette score for chosen k={best_k}: {sample_silhouette:.4f}")
    # create per-cluster silhouette average (not full plot)
    sil_per_cluster = []
    for i in range(best_k):
        mask = labels == i
        if np.sum(mask) > 1:
            sil_per_cluster.append(silhouette_score(X_scaled[mask], labels[mask]) if False else None)
        else:
            sil_per_cluster.append(np.nan)
    # Save summary
    summary = {
        'best_k': best_k,
        'silhouette_score': float(sample_silhouette),
        'n_features': int(n_features)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, 'kmeans_summary.csv'), index=False)
    print("Saved kmeans summary to", os.path.join(OUT_DIR, 'kmeans_summary.csv'))
except Exception as e:
    print("Could not compute detailed silhouette stats:", e)

# ---------- Save inertia and silhouette tables ----------
pd.DataFrame({'k': ks, 'inertia': inertias}).to_csv(os.path.join(OUT_DIR, 'elbow_inertia.csv'), index=False)
pd.DataFrame({'k': ks_sil, 'silhouette': sil_scores}).to_csv(os.path.join(OUT_DIR, 'silhouette_scores.csv'), index=False)
print("Saved inertia & silhouette csv files.")

print("\nAll outputs saved to:", OUT_DIR)
print("Done.")
