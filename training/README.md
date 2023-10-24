
# Pix2Struct Model Fine Tuning

This folder provides tools for fine-tuning  Google's Pix2struct model. It allows for interactive engagement with the training process, detailed visualization of steps, and the ability to finetune the model with custom datasets.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the necessary Python packages. At the root directory run the following command to install the packages:
  ```
  pip install -r requirements.txt
  ```
## Quick Start

### Interactive Training with Jupyter Notebook

1. Launch Jupyter notebook by running or imported at Google Colaboratory:
   ```
   jupyter notebook train_model.ipynb
   ```
2. The notebook is self-explanatory and guides you through the training process. Also, it is heavily influced from 

### Automated Training with Python Script

The Python script is for a more hands-off approach and well suited for environments equipped with a GPU.

1. Run the script using the following command:
   ```
   python train_model.py
   ```

#### Adding Your Datasets for Finetuning

The script uses the `hk-kaden-kim/pix2struct-chartcaptioning` dataset from Hugging Face by default. To finetune the model with your datasets, replace this with your dataset in the `load_dataset()` function call in the script.

## Model Training Details

- The script and notebook are configured to use the `google/pix2struct-base` model from Hugging Face as the starting point.
- The training includes an early stopping mechanism to prevent overfitting.

## Saving the Model

After training, the model is saved to a directory specified in the script/notebook. Make sure the directory exists and you have the necessary write permissions.
