import os
import shutil

# --- Variabili d'ambiente per ottimizzazioni Unsloth (devono essere impostate PRIMA di importare unsloth/torch/transformers) ---
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["UNSLOTH_SKIP_TRITON_PATCH"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# --- Import Unsloth PRIMA di torch/transformers ---
import unsloth
from unsloth import FastLanguageModel

# --- Ora tutti gli altri import ---
import torch
import gc
import time
import subprocess
import threading
import queue
import matplotlib.pyplot as plt
from huggingface_hub import login
import pandas as pd
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from trl import SFTTrainer
import numpy as np

# =====================
# CONFIGURAZIONE UTENTE
# =====================

# Hugging Face Token: da gestire come variabile d'ambiente!!!

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
        pass # Non stampare errori se CUDA non è disponibile o ci sono altri problemi
    return 0.0, 0.0 # Valori di default se CUDA non è disponibile o errore

def vram_monitor_thread(interval, data_queue):
    """Thread che monitora periodicamente l'uso della VRAM."""
    while not stop_vram_monitor.is_set():
        allocated, reserved = get_vram_usage()
        data_queue.put((time.time(), allocated, reserved))
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

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.log_history:
            return control # Nessun log ancora, continua

        current_eval_loss = None
        for log in reversed(state.log_history):
            if "eval_loss" in log:
                current_eval_loss = log["eval_loss"]
                break

        if current_eval_loss is None:
            return control # Nessuna eval_loss registrata in questo log, continua

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
vram_monitor_thread_handle = threading.Thread(target=vram_monitor_thread, args=(VRAM_MONITOR_INTERVAL, vram_data_queue))
vram_monitor_thread_handle.daemon = True
vram_monitor_thread_handle.start()
print(f"Monitoraggio VRAM avviato con intervallo di {VRAM_MONITOR_INTERVAL} secondi.")

# --- Misure di Tempo Globali ---
start_overall_time = time.time()

# --- Model Loading ---
print("\n===== Loading Model =====")
start_time = time.time()
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
        token=True,
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
        filtered_dataset = filtered_dataset.select(range(min(MAX_OBSERVATIONS, len(filtered_dataset))))
        print(f"DEBUG: Using only {len(filtered_dataset)} samples (MAX_OBSERVATIONS={MAX_OBSERVATIONS})")
    else:
        print(f"Using {len(filtered_dataset)} samples after filtering and shuffling.")

    split_dataset = filtered_dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

except Exception as e:
    print(f"Error loading or preparing dataset: {e}")
    print("Using minimal dummy data for both train and validation.")
    dummy_data = {
        "Sentence": ["I feel happy", "This makes me sad", "I'm excited", "Feeling calm today", "I am angry about this",
                     "That's fantastic!", "So disappointed", "Feeling joyful", "Quite relaxed", "Furious at the news"],
        "Label": ["happy", "sad", "excited", "calm", "angry", "happy", "sad", "happy", "calm", "angry"],
        "Trainset": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Debug: batch piccolo (originale: 8)
    gradient_accumulation_steps=1,  # Debug: update ogni batch (originale: 4)
    warmup_ratio=0.1,
    max_steps=4000,  #10 # Debug: pochi step (originale: 4000)
    learning_rate=2e-5,
    fp16=True,
    logging_steps=5,  
    output_dir=OUTPUT_DIR,
    optim="paged_adamw_8bit",
    save_strategy="no",
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    model=model,
    # RIMOSSO: tokenizer=tokenizer. Questo è il punto chiave.
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    evaluation_strategy="steps",
    eval_steps=20,
    callbacks=[CustomEarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE)],
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

# --- Merge LoRA weights into base model ---
print("\n===== Merging LoRA weights into base model =====")
if hasattr(FastLanguageModel, "merge_and_unload"):
    model = FastLanguageModel.merge_and_unload(model)
elif hasattr(FastLanguageModel, "merge_lora_weights"):
    model = FastLanguageModel.merge_lora_weights(model)
else:
    print("[ATTENZIONE] La tua versione di Unsloth non supporta il merge LoRA automatico. L'export GGUF potrebbe non funzionare.")

# --- Stop VRAM Monitor Thread and Collect Data ---
stop_vram_monitor.set()
vram_monitor_thread_handle.join()

while not vram_data_queue.empty():
    vram_data.append(vram_data_queue.get())

# --- Visualize Learning (Training Loss and Validation Loss) ---
print("\n===== Plotting Training and Validation Loss =====")
log_history = trainer.state.log_history
train_losses = []
train_steps = []
eval_losses = []
eval_steps = []

for log in log_history:
    if "loss" in log:
        train_losses.append(log["loss"])
        train_steps.append(log.get("step", 0))
    if "eval_loss" in log:
        eval_losses.append(log["eval_loss"])
        eval_steps.append(log.get("step", 0))

import numpy as np
plt.figure(figsize=(10, 6))
plotted = False

if train_losses and train_steps:
    plt.plot(train_steps, train_losses, label="Training Loss", color='blue', marker='o', markersize=4)
    plotted = True
else:
    print("[ATTENZIONE] Nessuna training loss trovata nei log. Il plot della training loss non verrà generato.")

if eval_losses and eval_steps:
    # Interpola la validation loss sui passi di training per un plot più fluido
    if len(eval_steps) > 1 and len(train_steps) > 1:
        eval_interp = np.interp(train_steps, eval_steps, eval_losses)
        plt.plot(train_steps, eval_interp, label="Validation Loss (interpolated)", color='red', linestyle='--', marker='x', markersize=4)
    else:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color='red', linestyle='--', marker='x', markersize=4)
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
    times = [d[0] for d in vram_data]
    allocated_vram = [d[1] for d in vram_data]
    reserved_vram = [d[2] for d in vram_data]

    start_time_plot = times[0]
    relative_times = [(t - start_time_plot) for t in times]

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

# --- Save Model (HuggingFace format) ---
print("\n===== Saving Model (HuggingFace format) =====")
try:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to '{OUTPUT_DIR}' (HuggingFace format). This includes the merged LoRA adapters.")
except Exception as e:
    print(f"Error saving model: {e}")

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
overall_execution_time = end_overall_time - start_overall_time
print(f"\nTempo complessivo di esecuzione dello script: {overall_execution_time:.1f} secondi") 