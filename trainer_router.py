import warnings
warnings.filterwarnings("ignore")

import time
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DistilBertModel, DistilBertTokenizer, Trainer, TrainingArguments, TrainerCallback
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

# Model Source: Can be a local path OR "distilbert-base-uncased"
MODEL_NAME = "distilbert-base-uncased" 

# JSONL file containing your 10k labeled samples
DATA_FILE = "sentinel_training_data_new.jsonl"

# Where checkpoints, logs, and final model are stored
OUTPUT_DIR = "./sentinel_router_v1"

# Max token length per sentence (Speed optimization)
MAX_LEN = 128


# ============================================================
# 1. MODEL ARCHITECTURE (Multi-Head + Dynamic Weights)
# ============================================================

class SentinelRouter(nn.Module):
    def __init__(self, base_model_name, pii_weight=1.0):
        super().__init__()
        
        # Load Base Model
        # Note: We handle local_files_only logic in the main execution block
        self.bert = DistilBertModel.from_pretrained(base_model_name)
        
        # Store the calculated weight for PII loss
        self.pii_weight = pii_weight
        
        # Head 1: Complexity (Regression -> 0.0 to 1.0)
        self.complexity_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Head 2: PII Detection (Classification -> Safe vs Sensitive)
        self.pii_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        complexity_score = self.complexity_head(cls_token)
        pii_logits = self.pii_head(cls_token)
        
        loss = None
        if labels is not None:
            # LOSS 1: Regression (MSE) for Complexity
            loss_fct_regr = nn.MSELoss()
            loss_complexity = loss_fct_regr(
                complexity_score.view(-1),
                labels["complexity_score"].view(-1)
            )
            
            # LOSS 2: Classification (CrossEntropy) for PII
            # We use the dynamic weight calculated from your dataset
            device = labels["pii_label"].device
            class_weights = torch.tensor([1.0, self.pii_weight]).to(device)
            
            loss_fct_cls = nn.CrossEntropyLoss(weight=class_weights)
            loss_pii = loss_fct_cls(
                pii_logits.view(-1, 2),
                labels["pii_label"].view(-1)
            )
            
            # Combined Loss
            loss = loss_complexity + loss_pii
        
        return {
            "loss": loss,
            "complexity": complexity_score,
            "pii_logits": pii_logits
        }


# ============================================================
# 2. DATASET LOADER
# ============================================================

class RouterDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": {
                "complexity_score": torch.tensor(
                    item["label"]["complexity_score"], dtype=torch.float
                ),
                "pii_label": torch.tensor(
                    int(item["label"]["contains_pii"]), dtype=torch.long
                )
            }
        }


# ============================================================
# 3. HELPERS (Collator & Callback)
# ============================================================

def custom_data_collator(features):
    batch = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": {
            "complexity_score": torch.stack([f["labels"]["complexity_score"] for f in features]),
            "pii_label": torch.stack([f["labels"]["pii_label"] for f in features])
        }
    }
    return batch

class ProgressLoggingCallback(TrainerCallback):
    def __init__(self, log_every=100):
        self.log_every = log_every
        self.step_start_time = None
        self.last_log_step = 0
        self.best_eval_loss = float('inf')

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.log_every == 0:
            self.step_start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % self.log_every == 0:
            elapsed = time.time() - self.step_start_time if self.step_start_time else 0
            loss = logs.get('loss', 0)
            print(f"   📊 Step {state.global_step} | Loss: {loss:.4f} | ⏱️ {elapsed:.2f}s")
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get('eval_loss', None)
            if eval_loss and eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                print(f"   🏆 New Best Validation Loss: {eval_loss:.6f}")


# ============================================================
# 4. MAIN TRAINING LOOP
# ============================================================

def train():
    print(f"🚀 Starting Sentinel-Router Training Pipeline...")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ Could not find {DATA_FILE}. Did you run the generation script?")
        
    print(f"📦 Loading dataset from {DATA_FILE}...")
    raw_data = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            raw_data.append(json.loads(line))
            
    # 2. Split Data
    train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42)
    print(f"   Train Size: {len(train_data)} | Val Size: {len(val_data)}")

    # 3. Calculate Class Weights Dynamically
    # This prevents the "Hardcoded Risk" - works for any dataset balance
    print("⚖️ Calculating Class Weights...")
    pii_counts = [int(x['label']['contains_pii']) for x in train_data]
    num_neg = pii_counts.count(0)
    num_pos = pii_counts.count(1)
    
    # Formula: num_negative / num_positive
    # Example: 1000 safe, 100 pii -> weight = 10.0 (PII errors cost 10x more)
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"   Safe Samples: {num_neg} | PII Samples: {num_pos}")
    print(f"   Calculated PII Weight: {pos_weight:.4f}")

    # 4. Initialize Tokenizer & Model
    print("🔧 Initializing Tokenizer & Model...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Pass the calculated weight to the model
    model = SentinelRouter(MODEL_NAME, pii_weight=pos_weight)

    train_dataset = RouterDataset(train_data, tokenizer, MAX_LEN)
    val_dataset = RouterDataset(val_data, tokenizer, MAX_LEN)

    # 5. Define Custom Trainer
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs, labels=labels)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        remove_unused_columns=False, # Critical for custom datasets
        report_to="none" # Disable wandb/mlflow for local run
    )

    if torch.backends.mps.is_available():
        print("⚡ Using Apple Silicon (MPS) Acceleration!")
    elif torch.cuda.is_available():
        print("⚡ Using CUDA GPU Acceleration!")
    else:
        print("🐢 Using CPU (This might be slow)...")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        callbacks=[ProgressLoggingCallback(log_every=50)]
    )

    # 7. Execute
    print("🔥 Training Started...")
    trainer.train()

    print(f"✅ Training Complete. Saving to {OUTPUT_DIR}...")
    
    # Save State Dict (Model weights)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pt")
    # Save Tokenizer (Vocabulary)
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")
    

if __name__ == "__main__":
    train()