import pandas as pd
import numpy as np
import torch
import os
import time
import gc
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel

# --- Global Configuration Parameters ---
# Path to your Parquet dataset file
INPUT_FILE_PATH = ('/home/dario/Downloads/LLMFT4STATS/emotions_dataset.parquet')
# Name of the column containing the text sentences
SENTENCE_COLUMN = 'Sentence'
# Name of the column containing the emotion labels (kept for data validation, though not used for embeddings directly)
LABEL_COLUMN = 'Label'
# Batch size for generating embeddings (adjust based on your GPU VRAM)
BATCH_SIZE = 128
# Number of rows to process for debugging (set to 0 to process all rows)
DEBUG_ROWS = 0  # Set to a small number (e.g., 1000) for quick tests
# Interval for clearing CUDA cache (in batches processed)
CLEAR_CACHE_INTERVAL = 10
# Output file path for the DataFrame with embeddings
OUTPUT_DF_PATH = '/home/dario/Downloads/LLMFT4STATS/emotions_dataset_with_embeddings_BGE.parquet'
# Model name for FlagEmbedding (BGE-M3 is recommended for general purpose)
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'


def generate_embeddings():
    """
    Generates sentence embeddings using BGE-M3 and saves the DataFrame with embeddings.
    """
    start_total_time = time.time()

    # --- 0. Environment Setup and Device Check ---
    print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
        use_fp16 = True  # Use float16 for reduced memory and faster computation on modern GPUs
    else:
        print(
            "WARNING: CUDA is not available. Embedding generation will run on CPU, which will be significantly slower.")
        device = "cpu"
        use_fp16 = False

    # --- 1. Load Dataset ---
    print(f"\nLoading data from {INPUT_FILE_PATH}...")
    try:
        df = pd.read_parquet(INPUT_FILE_PATH)
        if DEBUG_ROWS > 0:
            df = df.head(DEBUG_ROWS)  # For debugging, process only a subset

        print(f"Successfully loaded {len(df)} rows from the dataset.")

        # Display column names and first few rows for verification
        print("\nDataFrame Columns:")
        print(df.columns.tolist())
        print("\nFirst 5 rows of the DataFrame:")
        print(df.head())

        # Validate required columns exist
        if SENTENCE_COLUMN not in df.columns:
            print(f"Error: Sentence column '{SENTENCE_COLUMN}' not found in the input file.")
            return
        if LABEL_COLUMN not in df.columns:
            print(f"Error: Label column '{LABEL_COLUMN}' not found in the input file.")
            # This is not critical for embedding generation, but good to warn if it's expected
            print(f"Warning: Label column '{LABEL_COLUMN}' not found. It will not affect embedding generation.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    # Prepare texts for encoding: ensure they are strings and handle potential NaN values
    df['text_to_encode'] = df[SENTENCE_COLUMN].fillna("").astype(str)

    # Filter out empty sentences
    initial_rows = len(df)
    df = df[df['text_to_encode'].str.len() > 0].copy()  # Use .copy() to avoid SettingWithCopyWarning
    print(f"Filtered {initial_rows - len(df)} rows with empty text content.")
    if len(df) == 0:
        print("No valid text remaining after filtering. Exiting.")
        return
    print(f"Using {len(df)} sentences for embedding generation.")

    # --- 2. Initialize Embedding Model ---
    print(f"\nInitializing BGE-M3 model ({EMBEDDING_MODEL_NAME})...")
    try:
        # BGEM3FlagModel is optimized for sentence and paragraph embeddings
        model = BGEM3FlagModel(EMBEDDING_MODEL_NAME,
                               use_fp16=use_fp16,  # Use float16 if on GPU
                               device=device)
        print(f"Model '{EMBEDDING_MODEL_NAME}' loaded successfully on {device}.")
    except Exception as e:
        print(f"Error: Model loading failed: {e}")
        print("Please ensure you have an active internet connection to download the model.")
        print("You might need to install the 'FlagEmbedding' library (`pip install FlagEmbedding`).")
        return

    # --- 3. Generate Embeddings ---
    print("\nGenerating embeddings...")
    embeddings_list = []
    # Convert DataFrame column to list for efficient iteration
    texts_to_encode = df['text_to_encode'].tolist()

    for i in tqdm(range(0, len(texts_to_encode), BATCH_SIZE), desc="Generating Embeddings"):
        batch_texts = texts_to_encode[i:i + BATCH_SIZE]

        # `encode` generates embeddings. `return_dense=True` for standard dense embeddings.
        outputs = model.encode(
            batch_texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        embeddings_list.append(outputs['dense_vecs'])

        # Periodically clear CUDA cache to prevent OOM errors on GPU
        if device == "cuda" and (i // BATCH_SIZE) % CLEAR_CACHE_INTERVAL == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if not embeddings_list:
        print("No embeddings were generated. Please check input data and batching logic.")
        return

    # Vertically stack all generated embedding arrays into a single NumPy array
    embeddings = np.vstack(embeddings_list)
    print(f"Embeddings generated with shape: {embeddings.shape}")

    # Final cleanup of GPU cache after embedding generation
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Add embeddings to the DataFrame. It's crucial that the order matches the original 'df'.
    df['embedding'] = list(embeddings)

    # Save the DataFrame with generated embeddings to a Parquet file
    df.to_parquet(OUTPUT_DF_PATH, index=False)  # index=False to not save the DataFrame index
    print(f"DataFrame with embeddings saved to: {OUTPUT_DF_PATH}")

    end_total_time = time.time()
    print(f"\nTotal execution time: {end_total_time - start_total_time:.2f} seconds")


if __name__ == "__main__":
    # Check if the input file exists before proceeding
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Error: Input file not found at '{INPUT_FILE_PATH}'. Please check the path and try again.")
    else:
        generate_embeddings()