
#!/usr/bin/env python3

import sys
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from matplotlib import pyplot as plt

# Load the fine-tuned model and processor
model_path = "/path/to/final_model.pth"
model = Pix2StructForConditionalGeneration.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained("google/pix2struct-base", is_vqa=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt", max_patches=1024).to(device)
    flattened_patches = inputs.flattened_patches
    attention_mask = inputs.attention_mask

    generated_ids = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=50)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def display_images_with_captions(dataset):
    fig = plt.figure(figsize=(18, 14))
    for i, example in enumerate(dataset):
        image = example["image"]
        generated_caption = generate_caption(image)
        fig.add_subplot(2, 3, i+1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Generated caption: {generated_caption}")
    plt.show()

if __name__ == "__main__":
    response = input("Would you like to provide a custom image path? (yes/no): ").strip().lower()
    if response == 'yes':
        image_path = input("Please enter the path to your image: ").strip()
        image = plt.imread(image_path)
        print(generate_caption(image))
    else:
        # Use the football dataset
        dataset = load_dataset("ybelkada/football-dataset", split="train")
        display_images_with_captions(dataset)
