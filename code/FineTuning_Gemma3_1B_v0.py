import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["UNSLOTH_SKIP_TRITON_PATCH"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# Must import first for Unsloth optimizations
import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, DatasetDict
import gc
import time
from huggingface_hub import login
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from trl import SFTTrainer
import matplotlib.pyplot as plt
import psutil # Per monitorare l'uso della memoria (VRAM)
import threading # Per il thread di monitoraggio della VRAM
import queue # Per comunicare i dati di VRAM tra thread

# --- VRAM Monitor ---
vram_data = [] # Lista per salvare i dati di VRAM
stop_vram_monitor = threading.Event() # Evento per fermare il thread di monitoraggio

def get_vram_usage():
    """Restituisce l'uso corrente della VRAM in MB."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return allocated, reserved
    except Exception as e:
        #print(f"Errore durante il recupero della VRAM: {e}")
        pass # Non stampare errori se CUDA non è disponibile o ci sono altri problemi
    return 0.0, 0.0 # Valori di default se CUDA non è disponibile o errore

def vram_monitor_thread(interval, data_queue):
    """Thread che monitora periodicamente l'uso della VRAM."""
    while not stop_vram_monitor.is_set():
        allocated, reserved = get_vram_usage()
        data_queue.put((time.time(), allocated, reserved))
        time.sleep(interval)

def print_vram_usage(stage=""):
    allocated, reserved = get_vram_usage()
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"[VRAM] {stage}: Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB, Total: {total:.1f}GB")

# --- Configuration ---
MAX_SEQ_LENGTH = 512  # Optimized for emotion classification
MODEL_NAME = "google/gemma-3-1b-it"
DATASET_PATH = "/home/dario/Downloads/LLMFT4STATS/emotions_dataset_doe.parquet"
OUTPUT_DIR = "emogemma_finetuned"
LORA_CONFIG = {
    "r": 8,
    "target_modules": ["q_proj", "v_proj"],
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "use_gradient_checkpointing": True,
}
EARLY_STOPPING_PATIENCE = 50 # New: Early stopping patience in steps
VRAM_MONITOR_INTERVAL = 1 # Intervallo di campionamento VRAM in secondi

# --- Custom Early Stopping Callback ---
class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best_loss = float('inf')
        self.bad_steps = 0
        self.best_step = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Check if log_history is not empty
        if not state.log_history:
            return control # No logs yet, continue training

        # Get training loss from last log entry
        current_loss = state.log_history[-1].get("loss")
        if current_loss is None:
            return control # No loss logged yet for this step, or it's not a training loss log

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.bad_steps = 0
            self.best_step = state.global_step
        else:
            self.bad_steps += 1

        if self.bad_steps >= self.patience:
            print(f"\nEarly stopping triggered at step {state.global_step}. "
                  f"No improvement in training loss for {self.patience} consecutive steps. "
                  f"Best training loss: {self.best_loss:.4f} at step {self.best_step}.")
            control.should_training_stop = True

        return control

# --- Authentication ---
try:
    login()  # Use existing HF token
    print("Hugging Face authentication successful")
except:
    print("Failed to authenticate. Please run 'huggingface-cli login' first")
    exit(1)

# --- Start VRAM Monitor Thread ---
vram_data_queue = queue.Queue()
vram_monitor_thread_handle = threading.Thread(target=vram_monitor_thread, args=(VRAM_MONITOR_INTERVAL, vram_data_queue))
vram_monitor_thread_handle.daemon = True # Permette al programma di uscire anche se il thread è ancora in esecuzione
vram_monitor_thread_handle.start()
print(f"Monitoraggio VRAM avviato con intervallo di {VRAM_MONITOR_INTERVAL} secondi.")

# --- Misure di Tempo Globali ---
start_overall_time = time.time()

# --- Model Loading ---
print("\n===== Loading Model =====")
start_time = time.time()
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

# --- Apply LoRA ---
print("\n===== Configuring LoRA =====")
model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA configured. Trainable params: {trainable_params:,}")
print(f"Config: {LORA_CONFIG}")
print_vram_usage("After LoRA")

# --- Dataset Preparation ---
print("\n===== Loading and Preparing Dataset =====")
try:
    # 1. Load Parquet dataset
    full_dataset = load_dataset("parquet", data_files={"train": DATASET_PATH}, split="train")
    print(f"Loaded full dataset from '{DATASET_PATH}': {len(full_dataset)} samples")

    # Filter for 'Trainset' == 1
    filtered_dataset = full_dataset.filter(lambda x: x["Trainset"] == 1)
    print(f"Filtered dataset (Trainset=1): {len(filtered_dataset)} samples")

    if len(filtered_dataset) == 0:
        raise ValueError("No samples found where 'Trainset' == 1. Check your dataset and filtering logic.")

    # Reduce dataset size for VRAM constraints (still applies to the filtered set)
    filtered_dataset = filtered_dataset.shuffle(seed=42).select(range(min(10000, len(filtered_dataset))))
    print(f"Using {len(filtered_dataset)} samples after filtering and size reduction.")

    # 2. Split into 90% Train, 10% Validation
    split_dataset = filtered_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"] # 'test' in train_test_split corresponds to validation
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

except Exception as e:
    print(f"Error loading or preparing dataset: {e}")
    print("Using minimal dummy data for both train and validation.")
    dummy_data = {
        "Sentence": ["I feel happy", "This makes me sad", "I'm excited", "Feeling calm today", "I am angry about this"],
        "Label": ["happy", "sad", "excited", "calm", "angry"],
        "Trainset": [1, 1, 1, 1, 1] # Ensure dummy data has Trainset=1
    }
    # Create a dummy DataFrame and save it as a parquet file temporarily
    import pandas as pd
    dummy_df = pd.DataFrame(dummy_data)
    dummy_parquet_path = "dummy_emotions.parquet"
    dummy_df.to_parquet(dummy_parquet_path, index=False)

    dummy_dataset = load_dataset("parquet", data_files={"train": dummy_parquet_path}, split="train")
    dummy_dataset = dummy_dataset.filter(lambda x: x["Trainset"] == 1) # Still apply filter for consistency

    split_dataset = dummy_dataset.train_test_split(test_size=0.2, seed=42) # Adjusting split for small dummy
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Using dummy train dataset: {len(train_dataset)} samples")
    print(f"Using dummy validation dataset: {len(eval_dataset)} samples")
    # Clean up the dummy parquet file
    if os.path.exists(dummy_parquet_path):
        os.remove(dummy_parquet_path)


# Formatting function
def formatting_func(examples):
    texts = []
    for sentence, label in zip(examples["Sentence"], examples["Label"]):
        # Ensure 'Label' column exists in your dataset schema, if it's 'label' or something else, adjust here.
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
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    max_steps=500,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=5,
    output_dir=OUTPUT_DIR,
    optim="paged_adamw_8bit",
    save_strategy="no",
    report_to="tensorboard",
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    evaluation_strategy="steps",  # MOVED HERE
    eval_steps=20,  # MOVED HERE
    callbacks=[CustomEarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE)],
)

print("Training configuration complete")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"Total steps: {training_args.max_steps}")
print(f"Early Stopping Patience (steps without train loss improvement): {EARLY_STOPPING_PATIENCE}")
print_vram_usage("Before training")

# --- Training ---
print("\n===== Starting Training =====")
start_train = time.time()
trainer.train()
train_time = time.time() - start_train
print(f"\nTraining completed in {train_time:.1f} seconds")
print_vram_usage("After training")

# --- Stop VRAM Monitor Thread and Collect Data ---
stop_vram_monitor.set()
vram_monitor_thread_handle.join() # Attendi che il thread termini

while not vram_data_queue.empty():
    vram_data.append(vram_data_queue.get())

# --- Visualize Learning (Training Loss) ---
print("\n===== Plotting Training Loss =====")
log_history = trainer.state.log_history
train_losses = []
steps = []

for log in log_history:
    if "loss" in log and "step" in log:
        train_losses.append(log["loss"])
        steps.append(log["step"])

if len(train_losses) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Salva il plot in un file PNG
    loss_plot_path = os.path.join(OUTPUT_DIR, "training_loss.png")
    plt.savefig(loss_plot_path)
    plt.close() # Chiudi la figura per liberare memoria
    print(f"Training loss plot saved to '{loss_plot_path}'.")
else:
    print("No training loss data to plot.")

# --- Plot VRAM Usage ---
print("\n===== Plotting VRAM Usage =====")
if vram_data and torch.cuda.is_available():
    # Estrai i timestamp e i dati di VRAM
    times = [d[0] for d in vram_data]
    allocated_vram = [d[1] for d in vram_data]
    reserved_vram = [d[2] for d in vram_data]

    # Normalizza i timestamp all'inizio dell'esecuzione
    start_time_plot = times[0]
    relative_times = [(t - start_time_plot) for t in times]

    plt.figure(figsize=(12, 7))
    plt.plot(relative_times, allocated_vram, label="Allocated VRAM (MB)", color='blue')
    plt.plot(relative_times, reserved_vram, label="Reserved VRAM (MB)", color='red', linestyle='--')
    plt.xlabel("Time (seconds since start)")
    plt.ylabel("VRAM Usage (MB)")
    plt.title("VRAM Usage Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Salva il plot della VRAM in un file PNG
    vram_plot_path = os.path.join(OUTPUT_DIR, "vram_usage.png")
    plt.savefig(vram_plot_path)
    plt.close() # Chiudi la figura
    print(f"VRAM usage plot saved to '{vram_plot_path}'.")
elif not torch.cuda.is_available():
    print("Skipping VRAM usage plot: CUDA is not available.")
else:
    print("No VRAM usage data to plot.")

# --- Save Model ---
print("\n===== Saving Model =====")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to '{OUTPUT_DIR}'")
print_vram_usage("After saving")

# --- Cleanup ---
del model, trainer, formatted_train_dataset, formatted_eval_dataset
gc.collect()
torch.cuda.empty_cache()
print("\n===== Cleanup Complete =====")
print_vram_usage("Final")

# --- Tempo Complessivo di Esecuzione ---
end_overall_time = time.time()
overall_execution_time = end_overall_time - start_overall_time
print(f"\nTempo complessivo di esecuzione dello script: {overall_execution_time:.1f} secondi")

print("Use this GGUF conversion command:")
print(f"python -m unsloth.export_gguf --model_path {OUTPUT_DIR} --quantization q4_k_m")