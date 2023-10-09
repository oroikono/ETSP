#!/usr/bin/env python3

import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import numpy as np
from huggingface_hub import login

# Define constants
# ROOT = "folder"
MAX_PATCHES = 2048
EPOCHS = 5000

login("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")

# Set environment variables for cache and temporary directories
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("SCRATCH"), "huggingface_cache", "transformers")
cache_dir_data_path = os.path.join(os.environ.get("SCRATCH"), "huggingface_cache", "datasets")
os.environ["HF_DATASETS_CACHE"] = cache_dir_data_path  # Set this environment variable
os.environ["TMPDIR"] = os.path.join(os.environ.get("SCRATCH"), "tmp")

# Create directories and their entire path structure if they don't exist
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(cache_dir_data_path, exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

print("TRANSFORMERS_CACHE:", os.environ["TRANSFORMERS_CACHE"])
print("HF_DATASETS_CACHE:", os.environ["HF_DATASETS_CACHE"])
print("TMPDIR:", os.environ["TMPDIR"])


# Load dataset
dataset = load_dataset("oroikon/chart_captioning", cache_dir=cache_dir_data_path)

# Define EarlyStopping class for monitoring training
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def trace_func(self, message):
        print(message)

# Define ImageCaptioningDataset for loading image captioning data
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], truncation=True, return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

# Load model and processor
cache_dir_model_path = os.path.join(os.environ.get("SCRATCH"), "huggingface_cache", "transformers")
processor = AutoProcessor.from_pretrained("google/pix2struct-base", is_vqa=False, cache_dir=cache_dir_model_path)
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base", cache_dir=cache_dir_model_path)

# Collator function to process batches
def collator(batch):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["text"] for item in batch]
    text_inputs = processor(text=texts, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=20)
    new_batch["labels"] = text_inputs.input_ids
    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
    return new_batch

# Data loaders
train_dataset_icd = ImageCaptioningDataset(dataset['train'], processor)
train_dataloader = DataLoader(train_dataset_icd, shuffle=True, batch_size=2, collate_fn=collator)
val_dataset_icd = ImageCaptioningDataset(dataset['validation'], processor)
val_dataloader = DataLoader(val_dataset_icd, shuffle=False, batch_size=2, collate_fn=collator)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# Function to evaluate the model on the validation set
def evaluate(model, val_dataloader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            labels = batch["labels"].to(device)
            flattened_patches = batch["flattened_patches"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
    model.train()
    return val_loss / len(val_dataloader)

# Training loop
print("Start Training")
early_stopping = EarlyStopping(patience=5, verbose=True)
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    for idx, batch in enumerate(train_dataloader):
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    val_loss = evaluate(model, val_dataloader, device)
    print(f'Validation Loss: {val_loss:.6f}')
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    if (epoch + 1) % 20 == 0:
        model.eval()
        predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)
        print(f"Predictions: {processor.batch_decode(predictions, skip_special_tokens=True)}")
        model.train()

# Save the final model
torch.save(model.state_dict(), "final_model.pth")
