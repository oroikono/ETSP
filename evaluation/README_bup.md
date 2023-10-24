## Model Evaluation

The `evaluate_model.ipynb` notebook allows you to evaluate the performance of the [finetuned pix2struct model](https://huggingface.co/oroikon/ft_pix2struct_chart_captioning) on a [specified dataset](https://huggingface.co/datasets/hk-kaden-kim/pix2struct-chartcaptioning). It provides an interactive environment for loading the model, evaluating it, and viewing the generated captions.

### Using the Evaluation Notebook

1. **Prepare Your Environment**:
   - Ensure you have Jupyter installed. If not, you can install it using pip:
     ```
     pip install notebook
     ```
    OR
   - Import the notebook in Google Collaboratory

3. **Launch the Notebook**:
   - Open your terminal and run:
     ```
     jupyter notebook evaluate_model.ipynb
     ```

4. **Load the Model**:
   - The notebook is set up to evaluate a [specific finetuned model](https://huggingface.co/oroikon/ft_pix2struct_chart_captioning). If you want to evaluate a different model, replace the model path or model identifier in the designated cell.

5. **Load the Dataset**:
   - By default, the notebook is configured to evaluate the model on a [given dataset](https://huggingface.co/oroikon/ft_pix2struct_chart_captioning). If you wish to use a different dataset for evaluation, modify the dataset loading cell accordingly.

6. **Run the Evaluation**:
   - Execute the notebook cells sequentially to load the model, perform the evaluation, and generate captions. The notebook likely includes cells that display the evaluation metrics, helping you understand the model's performance.

7. **Review Generated Captions**:
   - After evaluation, the model's predictions (generated captions) are saved in a CSV file, typically named `generated_captions.csv`. You can find this file in the notebook's working directory and review the captions to assess the model's performance qualitatively.


