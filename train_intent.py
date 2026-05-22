import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler
)
from torch.optim import AdamW


# -------------------------
# Config
# -------------------------
INTENTS_PATH   = "intents.json"
MODEL_OUT_DIR  = "models/intent_model"
EPOCHS         = 10
BATCH_SIZE     = 8
LEARNING_RATE  = 5e-5
MAX_LENGTH     = 64


# -------------------------
# 1. Load + flatten intents
# -------------------------
def load_training_data(path: str):
    with open(path, "r") as f:
        intents = json.load(f)

    texts  = []
    labels = []
    label_map = {}          # intent name  →  int id
    id_map    = {}          # int id       →  intent name

    for idx, (intent, data) in enumerate(intents.items()):
        label_map[intent] = idx
        id_map[idx]       = intent

        for example in data["examples"]:
            texts.append(example)
            labels.append(idx)

    return texts, labels, label_map, id_map


# -------------------------
# 2. PyTorch Dataset
# -------------------------
class IntentDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# -------------------------
# 3. Train
# -------------------------
def train():

    print("Loading training data...")
    texts, labels, label_map, id_map = load_training_data(INTENTS_PATH)
    num_labels = len(label_map)

    print(f"  Intents   : {num_labels}")
    print(f"  Examples  : {len(texts)}")

    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    encodings = tokenizer(
        texts,
        truncation    = True,
        padding       = True,
        max_length    = MAX_LENGTH
    )

    dataset    = IntentDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    print("\nLoading distilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels = num_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model.to(device)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    num_training_steps = EPOCHS * len(dataloader)

    scheduler = get_scheduler(
        "linear",
        optimizer          = optimizer,
        num_warmup_steps   = 0,
        num_training_steps = num_training_steps
    )

    # Training loop
    print("\nTraining...\n")
    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        for batch in dataloader:

            batch  = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss   = output.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch + 1}/{EPOCHS}  —  loss: {avg_loss:.4f}")

    # -------------------------
    # 4. Save model + maps
    # -------------------------
    out_dir = Path(MODEL_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "label_map.json", "w") as f:
        json.dump({"label_map": label_map, "id_map": id_map}, f, indent=2)

    print(f"\nModel saved to: {MODEL_OUT_DIR}")
    print("Training complete.")


if __name__ == "__main__":
    train()