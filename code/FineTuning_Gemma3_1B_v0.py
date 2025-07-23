import os
import shutil
import pandas as pd

# --- Variabili d'ambiente per ottimizzazioni Unsloth (devono essere impostate PRIMA di importare unsloth/torch/transformers) ---
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["UNSLOTH_SKIP_TRITON_PATCH"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# --- Import Unsloth PRIMA di torch/transformers ---
import unsloth
from unsloth import FastLanguageModel

# --- Explicitly set Matplotlib backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use('Agg') # Use 'Agg' for non-interactive plotting (good for servers/headless environments)
import matplotlib.pyplot as plt

# --- Ora tutti gli altri import ---
import torch
import gc
import time
import subprocess
import threading
import queue
from huggingface_hub import login
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from trl import SFTTrainer
import numpy as np

# =====================
# CONFIGURAZIONE UTENTE
# =====================

# Hugging Face Token
hf_token = ############ da gestire come variabile d'ambiente

# Nome modello base
MODEL_NAME = "google/gemma-3-1b-it"

# Path dataset
DATASET_PATH = "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_doe.parquet"

# Directory di output
OUTPUT_DIR = "emogemma_finetuned"

# Configurazione LoRA
LORA_CONFIG = {
    "r": 8,
    "target_modules": ["q_proj", "v_proj"],
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "use_gradient_checkpointing": True,
}

# Pazienza per l'early stopping (in steps)
EARLY_STOPPING_PATIENCE = 50

# Intervallo di campionamento VRAM in secondi
VRAM_MONITOR_INTERVAL = 1

# Lunghezza massima sequenza
MAX_SEQ_LENGTH = 128  # Debug: sequenza corta (originale: 256 o 512)

# Numero massimo di osservazioni da usare (None = tutto il dataset, es. 5000 per debug)
MAX_OBSERVATIONS = None  # Debug: usa solo 100 osservazioni (originale: None o 5000)

# --- VRAM Monitor ---
# Global variable to store the start time of the main script execution for VRAM plot
global_script_start_time = time.time()
vram_data = [] # Lista per salvare i dati di VRAM
stop_vram_monitor = threading.Event() # Evento per fermare il thread di monitoraggio

def get_vram_usage():
    """Restituisce l'uso corrente della VRAM allocata e riservata in MB."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return allocated, reserved
    except Exception as e:
        # Avoid printing errors if CUDA isn't available or other issues
        # print(f"Warning: Could not get VRAM usage - {e}")
        pass
    return 0.0, 0.0 # Valori di default se CUDA non è disponibile o errore

def vram_monitor_thread(interval, data_queue, script_start_time):
    """Thread che monitora periodicamente l'uso della VRAM."""
    while not stop_vram_monitor.is_set():
        current_time = time.time()
        allocated, reserved = get_vram_usage()
        # Store relative time from script start
        data_queue.put((current_time - script_start_time, allocated, reserved))
        time.sleep(interval)

def print_vram_usage(stage=""):
    """Stampa l'uso corrente della VRAM."""
    allocated, reserved = get_vram_usage()
    total = 0.0
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3 # Total VRAM in GB
    print(f"[VRAM] {stage}: Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB, Total: {total:.1f}GB")


# --- Custom Early Stopping Callback (Basata sulla Validation Loss) ---
class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best_loss = float('inf')
        self.bad_steps = 0
        self.best_step = 0
        self.eval_logs = [] # Store evaluation logs to ensure they are available for plotting

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Retrieve the latest evaluation loss from metrics
        current_eval_loss = kwargs.get("metrics", {}).get("eval_loss")

        if current_eval_loss is None:
            # Fallback to log_history if metrics not directly passed (less reliable)
            for log in reversed(state.log_history):
                if "eval_loss" in log:
                    current_eval_loss = log["eval_loss"]
                    break

        if current_eval_loss is None:
            print("[Early Stopping] No eval_loss found in metrics or logs for this evaluation step. Continuing.")
            return control # Nessuna eval_loss registrata in questo log, continua

        # Store the evaluation log for later plotting consistency
        self.eval_logs.append({"step": state.global_step, "eval_loss": current_eval_loss})


        if current_eval_loss < self.best_loss:
            self.best_loss = current_eval_loss
            self.bad_steps = 0
            self.best_step = state.global_step
            print(f"\n[Early Stopping] New best validation loss: {self.best_loss:.4f} at step {self.best_step}")
        else:
            self.bad_steps += 1
            print(f"\n[Early Stopping] No improvement for {self.bad_steps}/{self.patience} evaluation steps. Current eval_loss: {current_eval_loss:.4f}, Best eval_loss: {self.best_loss:.4f} at step {self.best_step}")

        if self.bad_steps >= self.patience:
            print(f"\nEarly stopping triggered at step {state.global_step}. "
                  f"No improvement in validation loss for {self.patience} consecutive evaluation steps. "
                  f"Best validation loss: {self.best_loss:.4f} at step {self.best_step}.")
            control.should_training_stop = True

        return control

# --- Authentication (Hugging Face) ---
try:
    login(token=hf_token)
    print("Hugging Face authentication successful")
except Exception as e:
    print(f"Failed to authenticate: {e}. Please check your token.")
    exit(1)

# --- Creazione della directory di output ---
if os.path.exists(OUTPUT_DIR):
    print(f"[INFO] Cancello la cartella di output esistente: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory '{OUTPUT_DIR}' ensured.")

# --- Start VRAM Monitor Thread ---
vram_data_queue = queue.Queue()
# Pass global_script_start_time to the monitor thread for relative timing
vram_monitor_thread_handle = threading.Thread(target=vram_monitor_thread, args=(VRAM_MONITOR_INTERVAL, vram_data_queue, global_script_start_time))
vram_monitor_thread_handle.daemon = True
vram_monitor_thread_handle.start()
print(f"Monitoraggio VRAM avviato con intervallo di {VRAM_MONITOR_INTERVAL} secondi.")

# --- Misure di Tempo Globali ---
# global_script_start_time is already set at the top

# --- Model Loading ---
print("\n===== Loading Model =====")
start_time = time.time()
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
        token=hf_token, # Use hf_token here
        device_map="auto",
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    print_vram_usage("After loading")
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    print("Assicurati che il modello esista e che le dipendenze siano installate (es. bitsandbytes).")
    exit(1)

# --- Apply LoRA ---
print("\n===== Configuring LoRA =====")
model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"LoRA configured. Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
print(f"Config: {LORA_CONFIG}")
print_vram_usage("After LoRA")

# --- Dataset Preparation ---
print("\n===== Loading and Preparing Dataset =====")
try:
    full_dataset = load_dataset("parquet", data_files={"train": DATASET_PATH}, split="train")
    print(f"Loaded full dataset from '{DATASET_PATH}': {len(full_dataset)} samples")

    filtered_dataset = full_dataset.filter(lambda x: x["Trainset"] == 1)
    print(f"Filtered dataset (Trainset=1): {len(filtered_dataset)} samples")

    if len(filtered_dataset) == 0:
        raise ValueError("No samples found where 'Trainset' == 1. Check your dataset and filtering logic.")

    filtered_dataset = filtered_dataset.shuffle(seed=42)
    if MAX_OBSERVATIONS is not None:
        if MAX_OBSERVATIONS > len(filtered_dataset):
            print(f"WARNING: MAX_OBSERVATIONS ({MAX_OBSERVATIONS}) is greater than available filtered samples ({len(filtered_dataset)}). Using all available.")
        filtered_dataset = filtered_dataset.select(range(min(MAX_OBSERVATIONS, len(filtered_dataset))))
        print(f"DEBUG: Using only {len(filtered_dataset)} samples (MAX_OBSERVATIONS={MAX_OBSERVATIONS})")
    else:
        print(f"Using {len(filtered_dataset)} samples after filtering and shuffling.")

    split_dataset = filtered_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

except Exception as e:
    print(f"Error loading or preparing dataset: {e}")
    print("Using minimal dummy data for both train and validation.")
    dummy_data = {
        "Sentence": ["I feel happy", "This makes me sad", "I'm excited", "Feeling calm today", "I am angry about this",
                     "That's fantastic!", "So disappointed", "Feeling joyful", "Quite relaxed", "Furious at the news",
                     "I feel happy", "This makes me sad", "I'm excited", "Feeling calm today", "I am angry about this",
                     "That's fantastic!", "So disappointed", "Feeling joyful", "Quite relaxed", "Furious at the news"], # Increased dummy data
        "Label": ["happy", "sad", "excited", "calm", "angry", "happy", "sad", "happy", "calm", "angry",
                  "happy", "sad", "excited", "calm", "angry", "happy", "sad", "happy", "calm", "angry"],
        "Trainset": [1]*20
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_parquet_path = "dummy_emotions.parquet"
    dummy_df.to_parquet(dummy_parquet_path, index=False)

    dummy_dataset = load_dataset("parquet", data_files={"train": dummy_parquet_path}, split="train")
    dummy_dataset = dummy_dataset.filter(lambda x: x["Trainset"] == 1)

    split_dataset = dummy_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Using dummy train dataset: {len(train_dataset)} samples")
    print(f"Using dummy validation dataset: {len(eval_dataset)} samples")
    if os.path.exists(dummy_parquet_path):
        os.remove(dummy_parquet_path)

# Formatting function
def formatting_func(examples):
    texts = []
    for sentence, label in zip(examples["Sentence"], examples["Label"]):
        text = f"""<start_of_turn>user
Identify emotion: {sentence}<end_of_turn>
<start_of_turn>model
{label}<end_of_turn>{tokenizer.eos_token}"""
        texts.append(text)
    return {"text": texts}

formatted_train_dataset = train_dataset.map(formatting_func, batched=True)
formatted_eval_dataset = eval_dataset.map(formatting_func, batched=True)
print("Datasets formatted")

# --- Training Setup ---
print("\n===== Training Configuration =====")
# Initialize the callback instance before passing it to the Trainer
early_stopping_callback = CustomEarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    max_steps=5000,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=5,
    output_dir=OUTPUT_DIR,
    optim="paged_adamw_8bit",
    save_strategy="steps",
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # *** FIX THIS LINE ***
    eval_strategy="steps", # Changed from evaluation_strategy to eval_strategy
    eval_steps=20,
    # To see validation loss, max_steps must be >= eval_steps
    # If using small max_steps for debug, adjust eval_steps accordingly (e.g., eval_steps=1 for max_steps=10)
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    # evaluation_strategy and eval_steps are now defined in TrainingArguments
    callbacks=[early_stopping_callback],
)

print("Training configuration complete")
print(f"Batch size (per device): {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"Max steps: {training_args.max_steps}")
print(f"Early Stopping Patience (eval steps without eval loss improvement): {EARLY_STOPPING_PATIENCE}")
print(f"Evaluation frequency: Every {trainer.args.eval_steps} steps")
print_vram_usage("Before training")

# --- Training ---
print("\n===== Starting Training =====")
start_train = time.time()
trainer.train()
train_time = time.time() - start_train
print(f"\nTraining completed in {train_time:.1f} seconds")
print_vram_usage("After training")

# --- Save Model (HuggingFace format) ---
print("\n===== Saving Model (HuggingFace format) =====")
try:
    # When save_strategy="end_of_training", trainer.save_model() is typically called.
    # However, explicitly calling it here ensures the LoRA adapter files are saved
    # in the expected structure for the merge step.
    # The FastLanguageModel's save_pretrained is designed to save the PEFT adapters correctly.
    model.save_pretrained(OUTPUT_DIR) # This saves the PEFT adapters
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model (LoRA adapters) saved to '{OUTPUT_DIR}' (HuggingFace format).")
except Exception as e:
    print(f"Error saving model (LoRA adapters): {e}")

# --- Extract and Save Loss Data to Parquet ---
print("\n===== Saving Training and Validation Loss Data to Parquet =====")
# Access log_history directly from trainer state
log_history = trainer.state.log_history
loss_data = []

# Also include evaluation logs collected by the callback for robustness
# This ensures eval_loss is present even if the main log_history is sparse for it
combined_logs = list(log_history) + list(early_stopping_callback.eval_logs)
# Deduplicate and prioritize actual log_history entries if steps overlap
# For simplicity, we'll just process combined_logs and sort later

# Collect all relevant data
temp_data_dict = {} # Use a dictionary to easily update/override for the same step
for log in combined_logs:
    step = log.get("step")
    if step is not None:
        if step not in temp_data_dict:
            temp_data_dict[step] = {"step": step}
        if "loss" in log:
            temp_data_dict[step]["training_loss"] = log["loss"]
        if "eval_loss" in log:
            temp_data_dict[step]["validation_loss"] = log["eval_loss"]

loss_data = list(temp_data_dict.values()) # Convert dictionary values back to a list

if loss_data:
    loss_df = pd.DataFrame(loss_data)
    loss_df = loss_df.sort_values(by="step").reset_index(drop=True)

    # Fill NaNs for plotting/analysis if a loss type wasn't present at a step
    # For a cleaner plot line, you might want to forward-fill or interpolate NaNs
    # For raw data, leaving them as NaN is often preferred.
    # Here, we will just ensure columns exist and fill with None if not present for a step
    # This is more robust as pd.DataFrame will handle missing keys by creating NaN columns.
    # The sort and reset_index is the main cleanup.

    parquet_output_path = os.path.join(OUTPUT_DIR, "training_evaluation_loss.parquet")
    loss_df.to_parquet(parquet_output_path, index=False)
    print(f"Training and validation loss data saved to '{parquet_output_path}'.")
else:
    print("[ATTENZIONE] Nessun dato di loss trovato nei log da salvare.")



# --- Merge LoRA weights into base model (con PEFT explicito) ---
print("\n===== Merging LoRA weights into base model (PEFT) =====")
try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Use the same token for loading base model
    base_model_id = MODEL_NAME # Use the variable from your config
    lora_path = OUTPUT_DIR        # cartella output del fine-tuning
    merged_path = OUTPUT_DIR + "_merged"

    # Carica il modello base
    model_base = AutoModelForCausalLM.from_pretrained(base_model_id, token=hf_token, torch_dtype=torch.float16) # Add dtype for consistency
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)

    # Carica i pesi LoRA
    model_peft = PeftModel.from_pretrained(model_base, lora_path)

    # Fonde i pesi LoRA nel modello base
    merged_model = model_peft.merge_and_unload()

    # Salva il modello fuso
    merged_model.save_pretrained(merged_path)
    tokenizer_base.save_pretrained(merged_path)
    print(f"Modello merged salvato in {merged_path}")
except Exception as e:
    print(f"[ERRORE] Merge LoRA esplicito fallito: {e}\nL'export GGUF potrebbe non funzionare. Assicurati che '{OUTPUT_DIR}' contenga i file degli adapter (adapter_config.json, adapter_model.safetensors) e che il modello base sia accessibile.")

# --- Stop VRAM Monitor Thread and Collect Data ---
stop_vram_monitor.set()
vram_monitor_thread_handle.join()

while not vram_data_queue.empty():
    vram_data.append(vram_data_queue.get())

# --- Visualize Learning (Training Loss and Validation Loss) ---
print("\n===== Plotting Training and Validation Loss =====")
# Re-read from the parquet file to ensure consistency, or use the generated loss_df
if 'loss_df' in locals() and not loss_df.empty: # Check if loss_df was successfully created
    train_losses = loss_df['training_loss'].dropna().tolist()
    train_steps = loss_df.loc[loss_df['training_loss'].notna(), 'step'].tolist()
    eval_losses = loss_df['validation_loss'].dropna().tolist()
    eval_steps = loss_df.loc[loss_df['validation_loss'].notna(), 'step'].tolist()
else:
    # Fallback to direct log_history if parquet saving failed
    train_losses = []
    train_steps = []
    eval_losses = []
    eval_steps = []
    for log in trainer.state.log_history:
        if "loss" in log:
            train_losses.append(log["loss"])
            train_steps.append(log.get("step", 0))
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])
            eval_steps.append(log.get("step", 0))
    # Also include the eval_logs from the callback to make sure all eval points are captured
    for log in early_stopping_callback.eval_logs:
        if log["step"] not in eval_steps: # Avoid duplicates if already in log_history
            eval_steps.append(log["step"])
            eval_losses.append(log["eval_loss"])
    # Sort after combining
    eval_combined = sorted(zip(eval_steps, eval_losses), key=lambda x: x[0])
    eval_steps = [s for s, _ in eval_combined]
    eval_losses = [l for _, l in eval_combined]


plt.figure(figsize=(10, 6))
plotted = False

if train_losses and train_steps:
    plt.plot(train_steps, train_losses, label="Training Loss", color='blue', marker='o', markersize=4, linestyle='-')
    plotted = True
else:
    print("[ATTENZIONE] Nessuna training loss trovata nei log. Il plot della training loss non verrà generato.")

if eval_losses and eval_steps:
    # Check if there are enough points for interpolation
    if len(eval_steps) > 0 and len(train_steps) > 0:
        # Interpolate evaluation loss only if it makes sense (eval_steps cover the range of train_steps)
        # Using marker='x' at eval_steps makes it clear where actual evaluations occurred
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color='red', linestyle='--', marker='x', markersize=6)
    plotted = True
else:
    print("[ATTENZIONE] Nessuna validation loss trovata nei log. Il plot della validation loss non verrà generato.")

if plotted:
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(OUTPUT_DIR, "training_validation_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Plot salvato in {loss_plot_path}")
else:
    print("[ATTENZIONE] Nessuna curva di loss disponibile da plottare. Nessun plot generato.")


# --- Plot VRAM Usage ---
print("\n===== Plotting VRAM Usage =====")
if vram_data and torch.cuda.is_available():
    # Times are already relative to global_script_start_time due to the updated monitor thread
    relative_times = [d[0] for d in vram_data]
    allocated_vram = [d[1] for d in vram_data]
    reserved_vram = [d[2] for d in vram_data]

    plt.figure(figsize=(12, 7))
    plt.plot(relative_times, allocated_vram, label="Allocated VRAM (MB)", color='blue')
    plt.plot(relative_times, reserved_vram, label="Reserved VRAM (MB)", color='red', linestyle='--')
    plt.xlabel("Time (seconds since script start)")
    plt.ylabel("VRAM Usage (MB)")
    plt.title("VRAM Usage Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    vram_plot_path = os.path.join(OUTPUT_DIR, "vram_usage.png")
    plt.savefig(vram_plot_path)
    plt.close()
    print(f"VRAM usage plot saved to '{vram_plot_path}'.")
elif not torch.cuda.is_available():
    print("Skipping VRAM usage plot: CUDA is not available.")
else:
    print("No VRAM usage data to plot (VRAM monitor might not have collected data).")




# --- Cleanup ---
del trainer
del model
del tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- Chiusura esplicita di file e sessioni ---
try:
    import atexit
    import glob
    # Chiudi eventuali file TensorBoard
    for tb_file in glob.glob(os.path.join(OUTPUT_DIR, 'events.out.tfevents.*')):
        try:
            f = open(tb_file)
            f.close()
        except Exception:
            pass
    # Chiudi eventuali sessioni TensorBoard (se usate)
    try:
        from tensorboard import program
        tb = program.TensorBoard()
        tb.server.stop()
    except Exception:
        pass
except Exception:
    pass

print("\n===== Cleanup Complete =====")
print_vram_usage("Final")

# --- Tempo Complessivo di Esecuzione ---
end_overall_time = time.time()
overall_execution_time = end_overall_time - global_script_start_time
print(f"\nTempo complessivo di esecuzione dello script: {overall_execution_time:.1f} secondi") 