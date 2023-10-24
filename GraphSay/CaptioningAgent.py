import io
import os
from matplotlib.figure import Figure
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


class Pix2Struct_Base:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("oroikon/ft_pix2struct_chart_captioning")
        self.model = Pix2StructForConditionalGeneration.from_pretrained("oroikon/ft_pix2struct_chart_captioning")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def plot_to_png(self, plot: Figure) -> io.BytesIO:
        buf = io.BytesIO()
        plot.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def plotly_to_matplotlib(self, plot: go.Figure) -> Figure:
        # Extract data from Plotly figure
        plot_data = plot.data[0]  # Assuming one trace
        x_data = plot_data['x']
        y_data = plot_data['y']

        # Recreate the plot using Matplotlib
        matplotlib_fig, ax = plt.subplots()
        ax.scatter(x_data, y_data)
        ax.set_title(plot.layout.title.text)
        ax.set_xlabel(plot.layout.xaxis.title.text)
        ax.set_ylabel(plot.layout.yaxis.title.text)

        return matplotlib_fig

    def generate_caption(self, plot) -> str:
        # Check if the input plot is a plotly graph_objects Figure
        if isinstance(plot, go.Figure):
            matplotlib_fig = self.plotly_to_matplotlib(plot)
            image_stream = self.plot_to_png(matplotlib_fig)
        else:  # we assume it's a matplotlib figure
            image_stream = self.plot_to_png(plot)

        image = Image.open(image_stream)

        # Preprocess the image and prepare it for input into your model.
        inputs = self.processor(images=image, return_tensors="pt", max_patches=2048).to(self.device)
        flattened_patches = inputs.flattened_patches
        attention_mask = inputs.attention_mask

        # Generate caption
        generated_ids = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=100)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_caption
