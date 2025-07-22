import pandas as pd

def process_and_sample_data(input_file_path, output_file_path, num_samples=50000, random_seed=42, min_validation_per_group=10, min_group_size_for_validation=20):
    """
    Carica un dataset Parquet, processa colonne specifiche, esegue un campionamento stratificato
    basato sulle combinazioni "Label" e "Umap10KMeans", e salva il risultato.
    Garantisce un numero minimo di osservazioni nel validation set per gruppi grandi.

    Args:
        input_file_path (str): Percorso al file Parquet di input.
        output_file_path (str): Percorso dove verrà salvato il file Parquet di output.
        num_samples (int): Il numero totale di osservazioni da campionare.
        random_seed (int): Seed per la riproducibilità del campionamento casuale.
        min_validation_per_group (int): Numero minimo di osservazioni da riservare per il validation set
                                         all'interno di un gruppo specifico.
        min_group_size_for_validation (int): Dimensione minima del gruppo affinché venga applicata
                                               la regola di min_validation_per_group.
    """
    # 1. Carica il file Parquet, mantenendo solo le colonne specificate
    print(f"Loading file: {input_file_path}")
    try:
        df = pd.read_parquet(input_file_path, columns=["Sentence", "Label", "cluster_kmeans_umap10d"])
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the Parquet file: {e}")
        return

    # 2. Rinomina la colonna "cluster_kmeans_umap10d" in "Umap10KMeans"
    print("Renaming 'cluster_kmeans_umap10d' to 'Umap10KMeans'...")
    df.rename(columns={"cluster_kmeans_umap10d": "Umap10KMeans"}, inplace=True)

    # Inizializza la colonna "Trainset" a 0 per tutte le osservazioni
    df["Trainset"] = 0

    # 3. Identifica le osservazioni con campionamento basato sulle combinazioni "Label" - "Umap10KMeans"
    print(f"Starting sampling of {num_samples} observations...")

    # Calcola la dimensione di ciascun gruppo (combinazione di "Label" e "Umap10KMeans")
    group_sizes = df.groupby(["Label", "Umap10KMeans"]).size().reset_index(name='group_size')

    # Unisci group_sizes al DataFrame originale per ottenere la dimensione del gruppo per ogni riga
    df = df.merge(group_sizes, on=["Label", "Umap10KMeans"], how="left")

    # Identifica gli indici delle osservazioni che DEVONO essere nel validation set
    # per garantire il numero minimo per i gruppi grandi
    validation_reserved_indices = pd.Index([])
    for name, group in df.groupby(["Label", "Umap10KMeans"]):
        if len(group) > min_group_size_for_validation:
            # Se il gruppo è abbastanza grande, riserviamo min_validation_per_group osservazioni
            # per il validation set. Queste non saranno campionabili per il Trainset.
            reserved = group.sample(n=min_validation_per_group, random_state=random_seed).index
            validation_reserved_indices = validation_reserved_indices.union(reserved)

    # DataFrame delle osservazioni disponibili per il campionamento (escludendo quelle riservate al validation)
    df_sampling_pool = df.drop(index=validation_reserved_indices)

    # Calcola il numero di coppie "Label" - "Umap10KMeans" distinte nel pool di campionamento
    num_distinct_pairs = df_sampling_pool.groupby(["Label", "Umap10KMeans"]).ngroups

    if num_distinct_pairs == 0 or df_sampling_pool.empty:
        print("No distinct 'Label' - 'Umap10KMeans' pairs or no observations left in sampling pool. Cannot perform stratified sampling.")
        # Salva il DataFrame con 'Trainset' tutti zeri (e 0 per le riservate) dato che non è avvenuto alcun campionamento significativo
        df.loc[validation_reserved_indices, "Trainset"] = 0 # Assicurati che siano 0
        df.drop(columns=['group_size'], inplace=True) # Rimuovi solo group_size, sampling_probability non è stata aggiunta
        df.to_parquet(output_file_path, index=False)
        print(f"File saved: {output_file_path} (No samples extracted for Trainset, some reserved for validation).")
        return

    # Ricalcola le dimensioni dei gruppi e le probabilità solo per il pool di campionamento
    group_sizes_sampling_pool = df_sampling_pool.groupby(["Label", "Umap10KMeans"]).size().reset_index(name='group_size_pool')
    df_sampling_pool = df_sampling_pool.merge(group_sizes_sampling_pool, on=["Label", "Umap10KMeans"], how="left")

    # Calcola la probabilità di campionamento per ciascuna riga nel pool: 1 / (num_distinct_pairs * group_size_pool)
    df_sampling_pool['sampling_probability'] = 1 / (num_distinct_pairs * df_sampling_pool['group_size_pool'])

    # Gestisci potenziali problemi con le probabilità
    if df_sampling_pool['sampling_probability'].isnull().any() or (df_sampling_pool['sampling_probability'] == 0).all():
        print("Warning: Some sampling probabilities are NaN or all probabilities are zero in the sampling pool.")
        print("Falling back to uniform random sampling to select the requested number of samples from the pool.")
        # Se le probabilità sono problematiche, torna a un campionamento casuale semplice dal pool
        if len(df_sampling_pool) <= num_samples:
            sampled_indices = df_sampling_pool.index
        else:
            sampled_indices = df_sampling_pool.sample(n=num_samples, replace=False, random_state=random_seed).index
    else:
        # Assicurati che il numero di campioni richiesto non superi il totale delle osservazioni nel pool
        actual_num_samples = min(num_samples, len(df_sampling_pool))

        if actual_num_samples == len(df_sampling_pool):
            sampled_indices = df_sampling_pool.index # Prendi tutto se i campioni richiesti >= righe totali nel pool
        else:
            sampled_indices = df_sampling_pool.sample(
                n=actual_num_samples,
                weights='sampling_probability',
                replace=False,  # Campionamento senza sostituzione
                random_state=random_seed # Assicura la riproducibilità
            ).index

    # 4. Aggiungi la colonna "Trainset", impostando 1 per le osservazioni campionate, 0 per le altre
    # Le osservazioni riservate al validation rimarranno con Trainset=0
    df.loc[sampled_indices, "Trainset"] = 1

    # Rimuovi le colonne temporanee utilizzate per i calcoli di campionamento
    df.drop(columns=['group_size'], inplace=True)
    if 'group_size_pool' in df.columns: # Potrebbe non esserci se num_distinct_pairs era 0 inizialmente
        df.drop(columns=['group_size_pool', 'sampling_probability'], inplace=True)

    # 5. Salva il dataset risultante come file Parquet
    print(f"Saving processed dataset to: {output_file_path}")
    try:
        df.to_parquet(output_file_path, index=False)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"An error occurred while saving the Parquet file: {e}")

# --- Configurazione ---
# Input File 1
INPUT_PARQUET_PATH = "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_clusters_kmeans_umap10d.parquet"
OUTPUT_PARQUET_PATH = "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_doe.parquet"
NUMBER_OF_SAMPLES = 50000
REPRODUCIBILITY_SEED = 42 # Puoi cambiare questo intero per un campione diverso ma comunque riproducibile

# Nuovi parametri per il requisito di validation set minimo
MIN_VALIDATION_OBS_PER_GROUP = 20
MIN_GROUP_SIZE_FOR_VALIDATION = 40

# --- Esegui la funzione ---
process_and_sample_data(
    input_file_path=INPUT_PARQUET_PATH,
    output_file_path=OUTPUT_PARQUET_PATH,
    num_samples=NUMBER_OF_SAMPLES,
    random_seed=REPRODUCIBILITY_SEED,
    min_validation_per_group=MIN_VALIDATION_OBS_PER_GROUP,
    min_group_size_for_validation=MIN_GROUP_SIZE_FOR_VALIDATION
)
