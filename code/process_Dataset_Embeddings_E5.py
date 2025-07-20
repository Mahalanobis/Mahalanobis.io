import pandas as pd
import numpy as np
import torch
import os
import time
import gc
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

# -----------------------------------------------
# Configurazione
# -----------------------------------------------

# Percorsi dei file
INPUT_FILE_PATH = ('/home/dario/Downloads/LLMFT4STATS/emotions_dataset.parquet')
OUTPUT_DF_PATH = '/home/dario/Downloads/LLMFT4STATS/emotions_dataset_with_embeddings_E5.parquet'

# Nomi delle colonne nel dataset
SENTENCE_COLUMN = 'Sentence'
LABEL_COLUMN = 'Label'  # Usata per il contesto, ma non direttamente per gli embeddings

# Parametri di elaborazione
BATCH_SIZE = 128  # Dimensione del batch per l'inferenza del modello
DEBUG_ROWS = 0  # Numero di righe da processare per il debug (0 = processa tutte le righe)

# Configurazione del Modello di Embeddings
# Specifica il modello SentenceTransformer da utilizzare. Il modello determina la dimensione dell'embedding generato.
EMBEDDING_MODEL_NAME = 'intfloat/e5-large-v2'

# -----------------------------------------------


def generate_embeddings():
    """
    Genera embeddings delle frasi utilizzando la libreria sentence-transformers
    e salva il DataFrame risultante con la nuova colonna 'embedding'.
    """
    start_total_time = time.time()

    # --- 1. Configurazione dell'Ambiente e Dispositivo ---
    # Controlla la disponibilità della GPU (CUDA) per l'elaborazione.
    print(f"PyTorch Version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"GPU Disponibile. Utilizzo del dispositivo: {torch.cuda.get_device_name(0)}")
    else:
        print("ATTENZIONE: GPU (CUDA) non disponibile. L'elaborazione avverrà sulla CPU.")

    # --- 2. Caricamento e Preparazione dei Dati ---
    print(f"\nCaricamento dati dal percorso: {INPUT_FILE_PATH}...")
    try:
        df = pd.read_parquet(INPUT_FILE_PATH)

        # Filtra le righe se DEBUG_ROWS è impostato
        if DEBUG_ROWS > 0:
            df = df.head(DEBUG_ROWS)

        print(f"Caricate {len(df)} righe dal dataset.")

        # Assicura che la colonna delle frasi esista e prepara il testo per l'encoding
        if SENTENCE_COLUMN not in df.columns:
            print(f"ERRORE: Colonna '{SENTENCE_COLUMN}' non trovata nel DataFrame.")
            return

        # Prepara il testo: gestisce valori mancanti e converte in stringhe
        df['text_to_encode'] = df[SENTENCE_COLUMN].fillna("").astype(str)

        # Rimuove le righe con testo vuoto
        initial_rows = len(df)
        df = df[df['text_to_encode'].str.len() > 0].copy()

        print(f"Rimosse {initial_rows - len(df)} righe con testo vuoto.")
        if len(df) == 0:
            print("Nessun testo valido da elaborare. Uscita.")
            return

        print(f"Processo avviato su {len(df)} frasi valide.")

    except FileNotFoundError:
        print(f"ERRORE: File di input non trovato in {INPUT_FILE_PATH}")
        return
    except Exception as e:
        print(f"ERRORE durante il caricamento dei dati: {e}")
        return

    # --- 3. Inizializzazione del Modello di Embeddings ---
    print(f"\nInizializzazione del modello '{EMBEDDING_MODEL_NAME}'...")
    try:
        # Carica il modello SentenceTransformer e lo sposta automaticamente sul dispositivo (GPU o CPU)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        print(f"Modello caricato con successo sul dispositivo '{device}'.")
    except Exception as e:
        print(f"ERRORE: Caricamento del modello fallito: {e}")
        return

    # --- 4. Generazione degli Embeddings ---
    print("\nGenerazione embeddings in corso...")

    # Estrae la lista delle frasi da codificare
    texts_to_encode = df['text_to_encode'].tolist()

    # Genera gli embeddings. La libreria gestisce il batching e l'utilizzo della GPU in modo efficiente.
    embeddings = model.encode(
        texts_to_encode,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,  # Visualizza l'avanzamento
        convert_to_numpy=True  # Restituisce i risultati come un array NumPy
    )

    print(f"Generati {embeddings.shape[0]} embeddings con {embeddings.shape[1]} dimensioni ciascuno.")

    # Aggiunge gli embeddings al DataFrame come una nuova colonna
    df['embedding'] = list(embeddings)

    # Pulizia della memoria (consigliato in ambiente GPU)
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # --- 5. Salvataggio del Risultato ---
    print(f"\nSalvataggio del DataFrame con embeddings in: {OUTPUT_DF_PATH}")

    # Salva il DataFrame finale in un file Parquet
    df.to_parquet(OUTPUT_DF_PATH, index=False)

    end_total_time = time.time()
    print(f"\nProcesso completato in {end_total_time - start_total_time:.2f} secondi.")


# Esecuzione principale
if __name__ == "__main__":
    # Verifica che il file di input esista prima di avviare il processo di generazione
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Errore: File di input non trovato: '{INPUT_FILE_PATH}'. Verifica il percorso.")
    else:
        generate_embeddings()