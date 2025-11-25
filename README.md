# K-Means Clustering – Task 8

This project demonstrates **unsupervised learning using K-Means clustering**.  
It includes dataset preprocessing, elbow method, silhouette analysis, PCA-based visualization, and saving clustering results.

---

## 📌 Objective
Perform unsupervised clustering using **K-Means** to segment data and evaluate cluster quality.

---

## 🛠 Tools & Libraries
- **Python 3**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Joblib**

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn joblib
📁 Project Structure
graphql
Copy code
kmeans_clustering_task8.py          # Main script for clustering
kmeans_outputs/                     # Auto-generated after running
│
├── elbow_inertia.png               # Elbow plot (Inertia vs K)
├── silhouette_vs_k.png             # Silhouette score plot
├── clusters_pca.png                # 2D PCA cluster visualization
├── silhouette_plot_per_sample.png  # Silhouette per-cluster visualization
├── data_with_clusters.csv          # Input data + cluster labels
├── k_inertia_silhouette.csv        # Inertia & silhouette for K range
├── kmeans_summary.csv              # Selected K + metrics
├── kmeans_model.joblib             # Saved trained K-Means model
└── scaler.joblib                   # Saved StandardScaler
🧠 What This Project Does
✔ 1. Loads and Cleans Dataset
Supports .csv or .xlsx

Removes:

Empty columns

Non-numeric columns

Unnamed columns

Constant columns

Fills missing numeric values using median

✔ 2. Scales Features
Uses StandardScaler() to normalize features before clustering.

✔ 3. Uses Elbow Method to Find Optimal K
Plots Inertia (SSE) vs K

Helps identify best K where curve starts flattening

✔ 4. Computes Silhouette Score
Measures how well clusters are formed

Higher = better separation between clusters

✔ 5. Trains Final K-Means Model
Uses best K (selected via silhouette or elbow method)

Assigns cluster labels

Saves cluster assignments in CSV

✔ 6. Visualizes Clustering
PCA to reduce features to 2D (if necessary)

Creates colored cluster scatter plot

Displays cluster centers
## How to Run the Script

Step 1 — Add your dataset  
Place a `.csv` or `.xlsx` file in the project folder  
or update the script:

RAW_PATH = "path/to/your/data.csv"

Step 2 — Run:

python kmeans_clustering_task8.py

Step 3 — View all results inside:

kmeans_outputs/

------------------------------------------------------------

## Output Files Explained

File | Description
-----|------------
elbow_inertia.png | Elbow Method plot for finding K  
silhouette_vs_k.png | Silhouette scores for K = 2–12  
clusters_pca.png | PCA-based cluster visualization (2D)  
silhouette_plot_per_sample.png | Silhouette plot per cluster  
data_with_clusters.csv | Dataset + assigned clusters  
k_inertia_silhouette.csv | Inertia + silhouette for all K  
kmeans_summary.csv | Final K selection summary  
kmeans_model.joblib | Saved K-Means model  
scaler.joblib | Saved StandardScaler  

------------------------------------------------------------

## What You Will Learn

- K-Means clustering  
- Feature scaling  
- Elbow method (Inertia)  
- Silhouette score evaluation  
- PCA visualization  
- Complete unsupervised ML pipeline  

------------------------------------------------------------

