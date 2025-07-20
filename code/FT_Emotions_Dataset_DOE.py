import pandas as pd

def process_and_sample_data(input_file_path, output_file_path, num_samples=50000, random_seed=42):
    """
    Loads a Parquet dataset, processes specified columns, performs stratified sampling
    based on "Label" and "Umap10KMeans" combinations, and saves the result.

    Args:
        input_file_path (str): Path to the input Parquet file.
        output_file_path (str): Path where the output Parquet file will be saved.
        num_samples (int): The total number of observations to sample.
        random_seed (int): Seed for reproducibility of the random sampling.
    """
    # 1. Load the Parquet file, keeping only specified columns
    print(f"Loading file: {input_file_path}")
    try:
        df = pd.read_parquet(input_file_path, columns=["Sentence", "Label", "cluster_kmeans_umap10d"])
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the Parquet file: {e}")
        return

    # 2. Rename the "cluster_kmeans_umap10d" column to "Umap10KMeans"
    print("Renaming 'cluster_kmeans_umap10d' to 'Umap10KMeans'...")
    df.rename(columns={"cluster_kmeans_umap10d": "Umap10KMeans"}, inplace=True)

    # Initialize the "Trainset" column to 0 for all observations
    df["Trainset"] = 0

    # 3. Identify observations with sampling based on "Label" - "Umap10KMeans" combinations
    print(f"Starting sampling of {num_samples} observations...")

    # Calculate the number of distinct "Label" - "Umap10KMeans" pairs
    num_distinct_pairs = df.groupby(["Label", "Umap10KMeans"]).ngroups

    if num_distinct_pairs == 0:
        print("No distinct 'Label' - 'Umap10KMeans' pairs found. Cannot perform stratified sampling.")
        # Save the DataFrame with 'Trainset' all zeros as no sampling occurred
        df.to_parquet(output_file_path, index=False)
        print(f"File saved: {output_file_path} (No samples extracted).")
        return

    # Calculate the size of each group (combination of "Label" and "Umap10KMeans")
    group_sizes = df.groupby(["Label", "Umap10KMeans"]).size().reset_index(name='group_size')

    # Merge group_sizes back to the original DataFrame to get the group size for each row
    df = df.merge(group_sizes, on=["Label", "Umap10KMeans"], how="left")

    # Calculate the sampling probability for each row: 1 / (num_distinct_pairs * group_size)
    df['sampling_probability'] = 1 / (num_distinct_pairs * df['group_size'])

    # Handle potential issues with probabilities (e.g., if group_size was 0, though unlikely with .size())
    if df['sampling_probability'].isnull().any() or (df['sampling_probability'] == 0).all():
        print("Warning: Some sampling probabilities are NaN or all probabilities are zero.")
        print("Falling back to uniform random sampling to select the requested number of samples.")
        # If probabilities are problematic, revert to simple random sampling
        if len(df) <= num_samples: # If the dataset is smaller than or equal to the desired sample size
            sampled_indices = df.index
        else:
            sampled_indices = df.sample(n=num_samples, replace=False, random_state=random_seed).index
    else:
        # Normalize weights, although pandas' sample method handles this automatically
        # Ensure that the requested number of samples does not exceed total observations
        actual_num_samples = min(num_samples, len(df))

        if actual_num_samples == len(df):
            sampled_indices = df.index # Take all if requested samples >= total rows
        else:
            sampled_indices = df.sample(
                n=actual_num_samples,
                weights='sampling_probability',
                replace=False,  # Sample without replacement
                random_state=random_seed # Ensures reproducibility
            ).index

    # 4. Add "Trainset" column, setting 1 for sampled observations, 0 for others
    df.loc[sampled_indices, "Trainset"] = 1

    # Remove the temporary columns used for sampling calculations
    df.drop(columns=['group_size', 'sampling_probability'], inplace=True)

    # 5. Save the resulting dataset as a Parquet file
    print(f"Saving processed dataset to: {output_file_path}")
    try:
        df.to_parquet(output_file_path, index=False)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"An error occurred while saving the Parquet file: {e}")

# --- Configuration ---
# Input File 1
INPUT_PARQUET_PATH = "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_clusters_kmeans_umap10d.parquet"
OUTPUT_PARQUET_PATH = "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_doe.parquet"
NUMBER_OF_SAMPLES = 50000
REPRODUCIBILITY_SEED = 42 # You can change this integer for a different but still reproducible sample

# --- Execute the function ---
process_and_sample_data(
    input_file_path=INPUT_PARQUET_PATH,
    output_file_path=OUTPUT_PARQUET_PATH,
    num_samples=NUMBER_OF_SAMPLES,
    random_seed=REPRODUCIBILITY_SEED
)