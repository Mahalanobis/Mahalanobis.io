from datasets import load_dataset
import pandas as pd
import os

output_dir = '/home/dario/Downloads/LLMFT4STATS/'
output_file_name = 'emotions_dataset.parquet'
output_path = os.path.join(output_dir, output_file_name)

os.makedirs(output_dir, exist_ok=True)
print(f"Scaricamento del dataset 'boltuix/emotions-dataset'...")
try:
    dataset = load_dataset("boltuix/emotions-dataset")
    # I dataset di Hugging Face sono tipicamente divisi in 'train', 'validation', 'test'.
    # Se ci sono più split (train, validation, test), uniscili:
    all_splits_df = []
    for split_name in dataset.keys():
        df_split = dataset[split_name].to_pandas()
        all_splits_df.append(df_split)

    df_full = pd.concat(all_splits_df, ignore_index=True)
    df_full = df_full.rename(columns={'text': 'Sentence', 'labels': 'Label'})

    print(f"Dataset scaricato e unito. Total rows: {len(df_full)}")
    print(f"Nomi colonne dopo rinomina: {df_full.columns.tolist()}")
    print(f"Prime 5 righe del dataset unito:")
    print(df_full.head())

    df_full.to_parquet(output_path, index=False)
    print(f"Dataset salvato come {output_path}")

except Exception as e:
    print(f"Errore durante il download o il salvataggio del dataset: {e}")
    print("Assicurati di aver installato la libreria 'datasets': pip install datasets")
    exit()