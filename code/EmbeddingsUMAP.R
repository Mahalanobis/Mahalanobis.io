# R script to load the data, calculate UMAP 2D Label centroids, and plot the UMAP projection with Label text using ggrepel.

rm(list=ls())
options(scipen = 10, digits = 10)
Sys.setenv(TZ='GMT')
Sys.setlocale("LC_ALL", "en_US.UTF-8")

require(data.table)
require(arrow)
require(ggplot2)
require(ggrepel) # Required for geom_text_repel

# Set working directory
wdir <- "/home/dario/Downloads/"
setwd( wdir )

# Load original dataset (dt) - optional for this plot
file_path_orig <- "/home/dario/Downloads/LLMFT4STATS/emotions_dataset.parquet"
dt <- as.data.table(read_parquet(file_path_orig))

# Load the clustered dataset with UMAP 2D coordinates (tmp)
file_path_clustered <- "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_clusters_kmeans_umap10d.parquet"
tmp <- as.data.table(read_parquet(file_path_clustered))

# Check columns to ensure umap_plot_dim1 and umap_plot_dim2 are present
names(tmp)

# --- Calculate Centroids for each Original Label ---
# Group by 'Label' and calculate the mean of the UMAP coordinates
centroids <- tmp[, .(
  centroid_x = mean(umap_plot_dim1, na.rm = TRUE),
  centroid_y = mean(umap_plot_dim2, na.rm = TRUE)
), by = Label]

ggplot2::ggplot(data=tmp, aes(x=umap_plot_dim1, y=umap_plot_dim2, colour=Label)) + 
  geom_point(alpha = 1) + 
  ggrepel::geom_text_repel(data = centroids,
                           aes(x = centroid_x, y = centroid_y, label = Label),
                           size = 5,
                           box.padding = 0.5,
                           max.overlaps = Inf, # Ensure all labels are placed
                           segment.color = 'black',
                           colour = "black") + # Optional: draw lines from text to centroid
  
  theme_bw() +
  labs(title = "UMAP 2D BGE-Embeddings Projection with Emotion Centroids") +
  xlab("\nUMAP Dim 1") + ylab("\nUMAP Dim 2")


doe = tmp[,.N,by=.(Label,cluster_kmeans_umap10d)]
doe1 = tmp[,.N,by=.(Label)]
doe2 = tmp[,.N,by=.(cluster_kmeans_umap10d)]

file_path_emb <- "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_with_embeddings_BGE.parquet"
ee <- as.data.table(read_parquet(file_path_emb))


