
# Dataset Preparation

This folder include jupyter notebooks for creating Train / Val / Test dataset. Dataset is sampled from the benchmark dataset of [Chart-to-Text:Paper](https://aclanthology.org/2022.acl-long.277/). For this project, only Statista data was used, and among them, only Line and Bar graphs were focused.

## Fetch the benchmark dataset
First, you need to fetch the benchmark dataset from here [Chart-to-Text:Code](https://github.com/vis-nlp/chart-to-text). Then, locate both `dataset_creation.ipynh` and `utils.py` to create minimized dataset which is going to be used for fine-tuning and evaluation.

## Sampling 1000 from the benchmark dataset

In `dataset_creation.ipynb`, you can find the process of how to generate Train / Val / Test dataset from the benchmark dataset and upload to HuggingFace for usage.

First, total 1000 chart data for each single column type and multi column type are sampled. And it is divided into Train / Val / Test with 7:1:2 ratio.

After then, based on the graph element conditions, matplotlib graph are generated wih this information and save into the folder __./huggingface_dataset__.

Finally, All data are merged into each folder Train / Val / Test. And then, upload into HuggingFace.

## Statistics of the loaded Dataset

In `dataset_stats.ipynb`, you can find how to load sampled dataset from HuggingFace and some exploratory data analysis (EDA) code for it. There are several analysis for captions, such as length analysis, word counts, and a named entity recognition (NER)