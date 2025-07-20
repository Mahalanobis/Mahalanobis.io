import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from cuml.manifold import UMAP
from cuml.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cudf

# --- Configuration ---
EMBEDDINGS_PATH = "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_with_embeddings_BGE.parquet"
OUTPUT_DIR = "/home/dario/Downloads/LLMFT4STATS/"
MODELS_DIR = os.path.join(OUTPUT_DIR, "Models")

# UMAP for K-Means Clustering (Higher Dimensions)
UMAP1_N_COMPONENTS_CLUSTERING = 10
UMAP1_N_NEIGHBORS_CLUSTERING = 200
UMAP1_MIN_DIST_CLUSTERING = 0.01

# UMAP for Visualization (2 Dimensions)
UMAP2_N_COMPONENTS_PLOT = 2
UMAP2_N_NEIGHBORS_PLOT = 15
UMAP2_MIN_DIST_PLOT = 0.1

# K-Means
KMEANS_N_CLUSTERS = 10
KMEANS_RANDOM_STATE = 42
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10
RANDOM_STATE_SAMPLING = 42
SAMPLE_SIZE_FOR_DISPLAY = 10 # Number of sentences to display if many are found

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)

print("--- Embeddings Clustering (GPU) ---")

# --- Caricamento degli embeddings ---
print(f"Caricamento degli embeddings da: {EMBEDDINGS_PATH}")
start_time = pd.Timestamp.now()
df = pd.read_parquet(EMBEDDINGS_PATH)
end_time = pd.Timestamp.now()
print(f"Dati caricati con successo in {(end_time - start_time).total_seconds():.2f} secondi.")

print(f"Numero di righe: {len(df)}")
print(f"Column names of input DataFrame: {df.columns.tolist()}")

# Ensure 'embedding' column exists
if 'embedding' not in df.columns:
    raise ValueError("DataFrame must contain an 'embedding' column.")

embeddings_np = np.stack(df['embedding'].values) # Stack to get a proper 2D array
print(f"Shape degli embeddings originali: {embeddings_np.shape}")
print(f"Tipo di dati degli embeddings originali: {embeddings_np.dtype}")

# Convert float16 to float32 for cuML compatibility if necessary
if embeddings_np.dtype == np.float16:
    print("Rilevato tipo di dati float16. Conversione a float32 per compatibilità GPU.")
    embeddings_np = embeddings_np.astype(np.float32)
    print(f"Tipo di dati degli embeddings dopo conversione: {embeddings_np.dtype}")

# --- Normalizzazione degli embeddings a lunghezza unitaria (L2-norm) ---
print("\n--- Normalizzazione degli embeddings a lunghezza unitaria (L2-norm) ---")
start_time = pd.Timestamp.now()
embeddings_normalized_np = normalize(embeddings_np, axis=1, copy=False)
end_time = pd.Timestamp.now()
print(f"Normalizzazione completata in {(end_time - start_time).total_seconds():.2f} secondi.")

# --- Trasferimento degli embeddings normalizzati a GPU (da NumPy a cuDF DataFrame) ---
print("\n--- Trasferimento degli embeddings normalizzati a GPU (da NumPy a cuDF DataFrame) ---")
start_time = pd.Timestamp.now()
embeddings_gdf = cudf.DataFrame(embeddings_normalized_np)
end_time = pd.Timestamp.now()
print(f"Trasferimento a GPU completato in {(end_time - start_time).total_seconds():.2f} secondi.")
print(f"Shape degli embeddings su GPU: {embeddings_gdf.shape} (tipo: {type(embeddings_gdf)})")

# --- Avvio della riduzione dimensionale con cuML UMAP (per Clustering) ---
print(f"\n--- Avvio della riduzione dimensionale con cuML UMAP a {UMAP1_N_COMPONENTS_CLUSTERING} componenti (per Clustering) ---")
print(f"  Parametri UMAP (Clustering): n_neighbors={UMAP1_N_NEIGHBORS_CLUSTERING}, min_dist={UMAP1_MIN_DIST_CLUSTERING}")
start_time = pd.Timestamp.now()
reducer_clustering = UMAP(n_neighbors=UMAP1_N_NEIGHBORS_CLUSTERING, min_dist=UMAP1_MIN_DIST_CLUSTERING,
                          n_components=UMAP1_N_COMPONENTS_CLUSTERING, random_state=KMEANS_RANDOM_STATE)
embeddings_reduced_clustering_gdf = reducer_clustering.fit_transform(embeddings_gdf)
end_time = pd.Timestamp.now()
print(f"Riduzione dimensionale cuML UMAP (Clustering) completata in {(end_time - start_time).total_seconds():.2f} secondi.")
print(f"Shape degli embeddings ridotti (Clustering) su GPU: {embeddings_reduced_clustering_gdf.shape} (tipo: {type(embeddings_reduced_clustering_gdf)})")

# Save UMAP model for Clustering
umap_clustering_model_path = os.path.join(MODELS_DIR, f"umap_reducer_for_clustering_{UMAP1_N_COMPONENTS_CLUSTERING}d.joblib")
joblib.dump(reducer_clustering, umap_clustering_model_path)
print(f"Modello UMAP (Clustering) salvato in: {umap_clustering_model_path}")

# --- Avvio della cluster analysis con cuML K-Means (su GPU) ---
print(f"\n--- Avvio della cluster analysis con cuML K-Means (su GPU) ---")
print(f"  Parametri K-Means: n_clusters={KMEANS_N_CLUSTERS}, random_state={KMEANS_RANDOM_STATE}, max_iter={KMEANS_MAX_ITER}, n_init={KMEANS_N_INIT}")
start_time = pd.Timestamp.now()
kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=KMEANS_RANDOM_STATE, max_iter=KMEANS_MAX_ITER, n_init=KMEANS_N_INIT)
kmeans.fit(embeddings_reduced_clustering_gdf)
cluster_ids_gdf = kmeans.predict(embeddings_reduced_clustering_gdf)
end_time = pd.Timestamp.now()
print(f"Cluster analysis con cuML K-Means completata in {(end_time - start_time).total_seconds():.2f} secondi.")

# Save K-Means model
kmeans_model_path = os.path.join(MODELS_DIR, f"kmeans_model_{UMAP1_N_COMPONENTS_CLUSTERING}d.joblib")
joblib.dump(kmeans, kmeans_model_path)
print(f"Modello K-Means salvato in: {kmeans_model_path}")

# --- Conversione dei cluster ID da GPU a CPU e aggiunta al DataFrame originale ---
print("\n--- Conversione dei cluster ID da GPU a CPU e aggiunta al DataFrame originale ---")
start_time = pd.Timestamp.now()
cluster_col_name = f'cluster_kmeans_umap{UMAP1_N_COMPONENTS_CLUSTERING}d'
df[cluster_col_name] = cluster_ids_gdf.to_pandas() # Convert cuDF Series to pandas Series
end_time = pd.Timestamp.now()
print(f"Trasferimento cluster ID a CPU e aggiunta al DataFrame completata in {(end_time - start_time).total_seconds():.2f} secondi.")

print("Prime 5 righe del DataFrame con i cluster ID:")
print(df[['Sentence', 'Label', cluster_col_name]].head())

# --- NEW: UMAP 2D Reduction for Plotting ---
print(f"\n--- NEW: Avvio della riduzione dimensionale con cuML UMAP a {UMAP2_N_COMPONENTS_PLOT} componenti (per Plotting) ---")
print(f"  Parametri UMAP (Plotting): n_neighbors={UMAP2_N_NEIGHBORS_PLOT}, min_dist={UMAP2_MIN_DIST_PLOT}")
start_time = pd.Timestamp.now()
reducer_plot = UMAP(n_neighbors=UMAP2_N_NEIGHBORS_PLOT, min_dist=UMAP2_MIN_DIST_PLOT,
                    n_components=UMAP2_N_COMPONENTS_PLOT, random_state=KMEANS_RANDOM_STATE) # Using KMEANS_RANDOM_STATE for consistency
embeddings_reduced_plot_gdf = reducer_plot.fit_transform(embeddings_gdf) # Use original normalized embeddings
end_time = pd.Timestamp.now()
print(f"Riduzione dimensionale cuML UMAP (Plotting) completata in {(end_time - start_time).total_seconds():.2f} secondi.")
print(f"Shape degli embeddings ridotti (Plotting) su GPU: {embeddings_reduced_plot_gdf.shape} (tipo: {type(embeddings_reduced_plot_gdf)})")

# Save 2D UMAP model
umap_plot_model_path = os.path.join(MODELS_DIR, f"umap_reducer_for_plot_{UMAP2_N_COMPONENTS_PLOT}d.joblib")
joblib.dump(reducer_plot, umap_plot_model_path)
print(f"Modello UMAP (Plotting) salvato in: {umap_plot_model_path}")

# Convert 2D reduced embeddings to pandas and add to DataFrame
embeddings_reduced_plot_np = embeddings_reduced_plot_gdf.to_pandas()
df[f'umap_plot_dim1'] = embeddings_reduced_plot_np.iloc[:, 0]
df[f'umap_plot_dim2'] = embeddings_reduced_plot_np.iloc[:, 1]
print("Prime 5 righe del DataFrame con i componenti UMAP 2D per il plot:")
print(df[[f'umap_plot_dim1', f'umap_plot_dim2']].head())

# --- FIX: Drop the 'embedding' column before saving to ensure Parquet compatibility ---
# The 'embedding' column can cause issues when reading the parquet file in R due to complex data types.
if 'embedding' in df.columns:
    print("\n--- Dropping 'embedding' column before saving for Parquet compatibility ---")
    df_to_save = df.drop(columns=['embedding'])
else:
    df_to_save = df

# --- Salvataggio del DataFrame finale con cluster ID e UMAP 2D ---
output_parquet_path = os.path.join(OUTPUT_DIR, f"emotions_dataset_clusters_kmeans_umap{UMAP1_N_COMPONENTS_CLUSTERING}d.parquet")
print(f"\n--- Salvataggio del DataFrame finale con cluster ID e UMAP 2D in: {output_parquet_path} ---")
start_time = pd.Timestamp.now()
# Save the new DataFrame 'df_to_save' which excludes the problematic 'embedding' column
df_to_save.to_parquet(output_parquet_path, index=False)
end_time = pd.Timestamp.now()
print(f"DataFrame finale salvato con successo in {(end_time - start_time).total_seconds():.2f} secondi.")
print("Processo di clustering completato.")


# --- Prepare data for plotting (cluster distribution) ---
# Ensure all cluster IDs are represented in the cross-tabulation (Counts)
# We use the original df for analysis plots as df_to_save only dropped the embedding column for saving
cross_tab_counts = pd.crosstab(df[cluster_col_name], df['Label'])
all_cluster_ids = np.arange(KMEANS_N_CLUSTERS)
cross_tab_counts_full = cross_tab_counts.reindex(all_cluster_ids, fill_value=0)

# Calculate proportions by row (each cluster ID sums to 1 or 100%)
cross_tab_proportions = cross_tab_counts_full.div(cross_tab_counts_full.sum(axis=1), axis=0) * 100
cross_tab_proportions = cross_tab_proportions.fillna(0) # Handle potential NaN if a cluster has 0 members

# --- Calculate Global Label Distribution ---
global_label_distribution = df['Label'].value_counts(normalize=True) * 100
all_labels_in_clusters = cross_tab_counts_full.columns
global_label_distribution = global_label_distribution.reindex(all_labels_in_clusters, fill_value=0)

# --- Calculate Deviation from Global Proportion ---
deviation_matrix = cross_tab_proportions.subtract(global_label_distribution, axis='columns')

# --- Generazione e salvataggio del grafico di distribuzione Cluster ID vs. Label (Counts) ---
print("\n--- Generazione e salvataggio del grafico di distribuzione Cluster ID vs. Label (Counts) ---")
plt.figure(figsize=(14, 10))
sns.heatmap(cross_tab_counts_full, annot=True, fmt="d", cmap="viridis", linewidths=.5)
plt.title(f'Counts of Original Labels within Each Cluster ID (K-Means on UMAP {UMAP1_N_COMPONENTS_CLUSTERING}D)')
plt.xlabel('Original Label')
plt.ylabel(f'Cluster ID (K-Means on UMAP {UMAP1_N_COMPONENTS_CLUSTERING}D)')
plt.tight_layout()
plot_path_counts = os.path.join(OUTPUT_DIR, "cluster_label_distribution_counts.png")
plt.savefig(plot_path_counts)
plt.close()
print(f"Count plot saved successfully to: {plot_path_counts}")

# --- Generazione e salvataggio del grafico di distribuzione Cluster ID vs. Label (Relative %) ---
print("\n--- Generazione e salvataggio del grafico di distribuzione Cluster ID vs. Label (Relative %) ---")
plt.figure(figsize=(14, 10))
sns.heatmap(cross_tab_proportions, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
plt.title(f'Relative Percentage of Original Labels within Each Cluster ID (K-Means on UMAP {UMAP1_N_COMPONENTS_CLUSTERING}D)')
plt.xlabel('Original Label')
plt.ylabel(f'Cluster ID (K-Means on UMAP {UMAP1_N_COMPONENTS_CLUSTERING}D)')
plt.tight_layout()
plot_path_percentages = os.path.join(OUTPUT_DIR, "cluster_label_distribution_percentages.png")
plt.savefig(plot_path_percentages)
plt.close()
print(f"Percentage plot saved successfully to: {plot_path_percentages}")

# --- Generazione e salvataggio del grafico di Deviazione dalla Distribuzione Globale ---
print("\n--- Generazione e salvataggio del grafico di Deviazione dalla Distribuzione Globale ---")
plt.figure(figsize=(14, 10))
# Using a divergent colormap for deviation (e.g., 'coolwarm', 'RdBu')
max_abs_deviation = deviation_matrix.abs().max().max()
sns.heatmap(deviation_matrix, annot=True, fmt=".1f", cmap="coolwarm", linewidths=.5,
            vmin=-max_abs_deviation, vmax=max_abs_deviation, center=0)
plt.title(f'Deviation from Global Label Distribution (Relative % by Cluster ID)\n(Positive = Over-represented, Negative = Under-represented)')
plt.xlabel('Original Label')
plt.ylabel(f'Cluster ID (K-Means on UMAP {UMAP1_N_COMPONENTS_CLUSTERING}D)')
plt.tight_layout()
plot_path_deviation = os.path.join(OUTPUT_DIR, "cluster_label_distribution_deviation.png")
plt.savefig(plot_path_deviation)
plt.close()
print(f"Deviation plot saved successfully to: {plot_path_deviation}")

# --- NEW: Generazione e salvataggio del grafico UMAP 2D colorato per Cluster ID ---
print(f"\n--- NEW: Generazione e salvataggio del grafico UMAP {UMAP2_N_COMPONENTS_PLOT}D colorato per Cluster ID ---")
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='umap_plot_dim1',
    y='umap_plot_dim2',
    hue=cluster_col_name, # Use the cluster IDs from the 10D UMAP K-Means
    palette='Spectral', # Good for categorical data
    data=df,
    s=10, # Marker size
    alpha=0.6, # Transparency
    legend='full'
)
plt.title(f'UMAP {UMAP2_N_COMPONENTS_PLOT}D Projection of Embeddings, Colored by K-Means Clusters ({UMAP1_N_COMPONENTS_CLUSTERING}D UMAP)')
plt.xlabel(f'UMAP {UMAP2_N_COMPONENTS_PLOT}D Dimension 1')
plt.ylabel(f'UMAP {UMAP2_N_COMPONENTS_PLOT}D Dimension 2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plot_path_umap_2d = os.path.join(OUTPUT_DIR, f"umap_{UMAP2_N_COMPONENTS_PLOT}d_clusters.png")
plt.savefig(plot_path_umap_2d)
plt.close()
print(f"UMAP {UMAP2_N_COMPONENTS_PLOT}D plot saved successfully to: {plot_path_umap_2d}")

# --- Esempio dei dati di cross-tabulation (prime 5 righe): ---
print("\n--- Esempio dei dati di cross-tabulation (prime 5 righe): ---")
print(cross_tab_counts_full.head())

# --- Esempio dei dati di proporzione (prime 5 righe): ---
print("\n--- Esempio dei dati di proporzione (prime 5 righe): ---")
print(cross_tab_proportions.head())

# --- Esempio dei dati di Deviazione (prime 5 righe): ---
print("\n--- Esempio dei dati di Deviazione dalla Distribuzione Globale (prime 5 righe): ---")
print(deviation_matrix.head())