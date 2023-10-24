
# Evaluation

This folder provides tools for fine-tuning  Google's Pix2struct model. It allows for interactive engagement with the training process, detailed visualization of steps, and the ability to finetune the model with custom datasets.

The `test_caption_gen.ipynb` and `evaluation.ipynb` notebook allows you to evaluate the performance of the finetuned pix2struct model ([HuggingFace:FTModel](https://huggingface.co/oroikon/ft_pix2struct_chart_captioning)) on the test dataset ([HuggingFace:Dataset](https://huggingface.co/datasets/hk-kaden-kim/pix2struct-chartcaptioning)).

## Generate Captions for Test Graphs

In `test_caption_gen.ipynb`, you can find not only how to load the finetuned model and test dataset from HuggingFace, but also generate captions for this test dataset. We highly recommend you to use GPU for the inference of all test dataset (1,400 graphs).

All generated captions and other information are saved into `test_caption_gen.csv` so as to be used for evaluation matrices.

## Evaluate generated captions

In `evaluation.ipynb`, you can find how to calculate main evaluation metrices from generated captions in `test_caption_gen.csv`. __BLEU__, __ROUGE__, and __METEOR__ are used for the evalution. The details for all these matrices can be found here.

- [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu)
- [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge)
- [METEOR](https://huggingface.co/spaces/evaluate-metric/meteor)
