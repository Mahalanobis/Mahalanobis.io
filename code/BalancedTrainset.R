
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

# Load original dataset (doe)
file_path_orig <- "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_doe.parquet"
dt <- as.data.table(read_parquet(file_path_orig))

mat = expand.grid( Label=sort(unique(dt$Label)) , Umap10KMeans= sort(unique(dt$Umap10KMeans)) )
mat = as.data.table(mat)
tmp = dt[Trainset==1,.N,by=.(Label,Umap10KMeans)]

mat = tmp[mat,on=c("Label","Umap10KMeans")]
mat[is.na(N),N:=0]
mat[,Umap10KMeans := as.character(Umap10KMeans)]

p <- ggplot(mat, aes(x = Umap10KMeans, y = Label, fill = log(N + 1) )) +
  geom_tile(color = "white") + # Aggiunge bordi bianchi alle celle per chiarezza
  geom_text(aes(label = N), color = "black", size = 3) + # Mostra il valore di N sulla cella
  scale_fill_gradient(low = "lightyellow", high = "red") + # Scala di colori per N
  labs(title = "Train-set",
       x = "\nUmap10KMeans Cluster",
       y = "Emoton Label\n",
       fill = "N") +
  theme_minimal() + # Un tema pulito per il plot
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + # Ruota le etichette dell'asse X se necessario
  coord_fixed() + # Mantiene le celle quadrate
  guides(fill="none") +
  theme_bw()

export_path <- "/home/dario/Downloads/Mahalanobis.io/docs/assets/images/"
file_name <- "trainset_n.png" # Puoi scegliere il nome del file e l'estensione (.png, .jpg, .pdf, .svg)
full_path <- file.path(export_path, file_name)
ggsave(filename = full_path, plot = p, width = 8, height = 6, units = "in", dpi = 300)



mat = expand.grid( Label=sort(unique(dt$Label)) , Umap10KMeans= sort(unique(dt$Umap10KMeans)) )
mat = as.data.table(mat)
tmp = dt[Trainset==0,.N,by=.(Label,Umap10KMeans)]

mat = tmp[mat,on=c("Label","Umap10KMeans")]
mat[is.na(N),N:=0]
mat[,Umap10KMeans := as.character(Umap10KMeans)]

p <- ggplot(mat, aes(x = Umap10KMeans, y = Label, fill = log(N + 1) )) +
  geom_tile(color = "white") + # Aggiunge bordi bianchi alle celle per chiarezza
  geom_text(aes(label = N), color = "black", size = 3) + # Mostra il valore di N sulla cella
  scale_fill_gradient(low = "cyan", high = "blue") + # Scala di colori per N
  labs(title = "Validation-set",
       x = "\nUmap10KMeans Cluster",
       y = "Emoton Label\n",
       fill = "N") +
  theme_minimal() + # Un tema pulito per il plot
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + # Ruota le etichette dell'asse X se necessario
  coord_fixed() + # Mantiene le celle quadrate
  guides(fill="none") +
  theme_bw()

export_path <- "/home/dario/Downloads/Mahalanobis.io/docs/assets/images/"
file_name <- "validset_n.png" # Puoi scegliere il nome del file e l'estensione (.png, .jpg, .pdf, .svg)
full_path <- file.path(export_path, file_name)
ggsave(filename = full_path, plot = p, width = 8, height = 6, units = "in", dpi = 300)

