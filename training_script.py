
#!/usr/bin/env python3

import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

# Parameters
root = "folder"
MAX_PATCHES = 1024
GRADIENT_ACCUMULATION_STEPS = 4

os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("SCRATCH"), "huggingface_cache", "transformers")


os.environ["TMPDIR"] = os.path.join(os.environ.get("SCRATCH"), "tmp")
if not os.path.exists(os.environ["TMPDIR"]):
    os.makedirs(os.environ["TMPDIR"])



# Load dataset
train_dataset = load_dataset("imagefolder", data_dir=root, split='train')
val_dataset = load_dataset("imagefolder", data_dir=root, split="validation")

# Dataset definition
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

# Model and processor
print("here")
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
processor = AutoProcessor.from_pretrained("google/pix2struct-base", is_vqa=False)
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")

print('passed')

# Collator function

def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
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
train_dataset_icd = ImageCaptioningDataset(train_dataset, processor)
train_dataloader = DataLoader(train_dataset_icd, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)

val_dataset_icd = ImageCaptioningDataset(val_dataset, processor)
val_dataloader = DataLoader(val_dataset_icd, shuffle=False, batch_size=2, collate_fn=collator, num_workers=4)

# Training setup
EPOCHS = 5000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# Gradient clipping value
clip_value = 1.0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for idx, batch in enumerate(train_dataloader):
        labels = batch["labels"].to(device)
        flattened_patches = batch["flattened_patches"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        if (idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

    print(f"Epoch: {epoch}, Loss: {epoch_loss/len(train_dataloader)}")

    # Checkpointing
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch}.pth")

# Save final model
torch.save(model.state_dict(), "final_model.pth")
