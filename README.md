# Essentials to Text and Speech Processing
References
- Dataset | Chart-to-Text:Paper](https://aclanthology.org/2022.acl-long.277/), [Chart-to-Text:Code](https://github.com/vis-nlp/chart-to-text)  
- Model | [Pix2Struct:Paper](https://www.semanticscholar.org/paper/Pix2Struct%3A-Screenshot-Parsing-as-Pretraining-for-Lee-Joshi/e1484706c0fab932fc9804df328044b3cb2f110d)
- Evaluation | [Chart-to-Text:Paper](https://aclanthology.org/2022.acl-long.277/), [Chart-to-Text:Code](https://github.com/vis-nlp/chart-to-text)  
- Overleaf | [FinalReport](https://www.overleaf.com/project/651c62028a734f8fcdff171a), [ProposalReport](https://www.overleaf.com/project/650770dd956cfc29aeca120d)


# Comprehensive Data Visualization and Model Training Repository

This repository is a one-stop resource for finetuning Google's Pix2struct model, evaluating its performance on a custom dataset derived from [Chart-to-Text:Paper](https://aclanthology.org/2022.acl-long.277/), and diving into charts' caption generation via GraphSay class, which forms the basis for an upcoming Chart-Captioning library.

## Repository Structure

- `dataset/`: Contains the datasets required for training and evaluating the model.
- `training/`: Includes resources and guidelines for finetuning  Google's Pix2struct model.
- `evaluation/`: Provides tools for evaluating the trained model and analyzing its performance.
- `library/`: Introduces the GraphSay class with use cases, setting the stage for a future chart captioning library.

## Training the Model

The `training/` directory contains all the necessary tools and instructions for training your pix2struct model. Detailed steps and requirements are outlined in the README within this directory. 

[Read more about Training](./training/README.md)

## Evaluating the Model

Once your model is trained, you can evaluate its performance using the resources in the `evaluation/` directory. The README in this section explains how to use the provided notebook for evaluation purposes.

[Read more about Evaluation](./evaluation/README.md)

## GraphSay: Towards a Comprehensive Visualization Library

The `library/` directory is dedicated to the GraphSay class, showcasing its capabilities and potential as the foundation for a future library. Explore various use cases supported by GraphSay in the interactive Python notebook provided.

[Read more about GraphSay](./library/README.md)

## Getting Started

To get started with this repository, clone it to your local machine and navigate to the specific directories for detailed guidelines and instructions. Ensure you meet the prerequisites for each section before proceeding with training, evaluation, or visualization tasks.

## Contribution

Contributions are welcome! Please read the contribution guidelines before starting any work.

## License

Distributed under the MIT License. See `LICENSE` for more information.

---


