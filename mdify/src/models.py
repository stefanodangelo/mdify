from mdify.src.utils import (
    CHART_DETECTION_MODEL_NAME, 
    CHART_DETECTION_MODEL_PATH, 
    CHART_DETECTION_N_CLASSES, 
    CHART_VQA_MODEL_NAME_CLASS, 
    CHART_VQA_PROMPT, 
    CHART_VQA_MODEL_CONFIG_PATH, 
    IMAGE_VQA_MODEL_NAME_CLASS, 
    IMAGE_VQA_MODEL_CONFIG_PATH,
    FORMULA_EXTRACTION_MODEL_NAME,
    open_image
)

from transformers import *
from optimum.onnxruntime import ORTModelForVision2Seq
from torchvision import transforms
from abc import abstractmethod
from typing import Iterable
from PIL import Image
import pandas as pd
import numpy as np
import torch
import yaml

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)  # Suppress Transformers' logs
logging.getLogger("datasets").setLevel(logging.ERROR)  # Suppress datasets' logs
logging.getLogger("torch").setLevel(logging.WARNING)  # Reduce PyTorch logs


class PictureClassifier:
    """
    Classifies images into categories such as "Picture" or "Chart" using a pretrained Swin Transformer model.

    Attributes:
        model (SwinForImageClassification): Pretrained Swin model for image classification.
        device (torch.device): Device to perform computations, either CPU or GPU.
        transform (torchvision.transforms.Compose): Preprocessing transformations for input images.
        id2class (dict): Mapping of class indices to class names.
    """

    def __init__(self):
        image_processor = AutoImageProcessor.from_pretrained(CHART_DETECTION_MODEL_NAME)

        # Create the model configuration without downloading
        config = AutoConfig.from_pretrained(CHART_DETECTION_MODEL_NAME, num_labels=CHART_DETECTION_N_CLASSES)
        self.model = SwinForImageClassification(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved weights
        self.model.load_state_dict(torch.load(CHART_DETECTION_MODEL_PATH, map_location=torch.device(self.device)))

        # Preprocessing transformations (same as during training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])

        self.id2class = {0: "Picture", 1: "Chart"}

    def classify(self, image: str | np.ndarray) -> int:
        """
        Classifies an input image into predefined categories, using GPU if available.

        Args:
            image (str | np.ndarray): Path to the image or an image in the form of a NumPy array.

        Returns:
            int: Predicted class index of the image.
        """

        # Load and preprocess a single image
        if type(image) == str:
            image = open_image(image, to_numpy=True)
        elif type(image) == np.ndarray:
            image = Image.fromarray(image)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor).logits  # Get logits from the model
            predicted_class = torch.argmax(outputs, dim=1).item()  # Get the class index

        return predicted_class

class HuggingFaceModel:
    """
    A generic wrapper for Hugging Face models that supports text and vision-based predictions.

    Attributes:
        processor: Preprocessor for the specific Hugging Face model.
        model: Pretrained Hugging Face model for predictions.
        config (dict): Additional configurations for model generation.
        prompt (str): Text prompt to be used during prediction (if applicable).
        use_pixel_values (bool): Whether to use pixel values instead of text tokens.
    """

    def __init__(self, huggingface_model_name: str, huggingface_class: str, config_path: str):
        self.processor = eval(f'{huggingface_class}Processor.from_pretrained(\'{huggingface_model_name}\')')
        self.model = eval(f'{huggingface_class}ForConditionalGeneration.from_pretrained(\'{huggingface_model_name}\')')
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.prompt = ''
        self.use_pixel_values = False

    def predict(self, image_path: str):
        """
        Generates a prediction for the input image using the Hugging Face model.

        Args:
            image_path (str): Path to the input image.

        Returns:
            The processed output from the model's predictions.
        """
        inputs = self.processor(images=Image.open(image_path), text=self.prompt, return_tensors="pt")
        preds = self.model.generate(**inputs, **self.config)
        pred_fn = 'decode'
        if self.use_pixel_values:
            preds = preds.pixel_values
            pred_fn = 'batch_' + pred_fn
        output = eval(f'self.processor.{pred_fn}(preds, skip_special_tokens=True)')
        return self._postprocess(output)
    
    @abstractmethod
    def _postprocess(self, text: str | Iterable[str]):
        """
        Postprocesses the raw output from the model. Must to be implemented by subclasses.

        Args:
            text (str | Iterable[str]): Raw output text from the model.

        Returns:
            Processed text or results.
        """
        pass


class ChartDeplotModel(HuggingFaceModel):
    """
    Specialized Hugging Face model for extracting data from charts.

    Attributes:
        placeholder (str): Placeholder token used in the model's output.
        separator (str): Separator used in the model's output.
    """

    def __init__(self):
        model, class_ = CHART_VQA_MODEL_NAME_CLASS
        super().__init__(model, class_, CHART_VQA_MODEL_CONFIG_PATH)
        self.placeholder = '<0x0A>'
        self.separator = '|'
        self.prompt = CHART_VQA_PROMPT

    def _postprocess(self, text: str):
        """
        Converts the model's raw output into a structured DataFrame.

        Args:
            text (str): Raw output text from the model.

        Returns:
            pandas.DataFrame: Extracted data structured in a tabular format.
        """

        # Split the cleaned string into lines
        lines = text.replace(self.placeholder, "\n").split("\n")

        # Extract the title and headers
        headers = [header.strip() for header in lines[1].split(self.separator)]

        # Extract the data rows
        data = []
        for line in lines[2:]:
            data.append([item.strip() for item in line.split(self.separator)])
        
        return pd.DataFrame(data, columns=headers)

        
class ImageCaptioningModel(HuggingFaceModel):
    """
    Hugging Face model for generating captions for images.
    """

    def __init__(self):
        model, class_ = IMAGE_VQA_MODEL_NAME_CLASS
        super().__init__(model, class_, IMAGE_VQA_MODEL_CONFIG_PATH)

    def _postprocess(self, text: str | Iterable[str]):
        """
        Processes the raw captions generated by the model.

        Args:
            text (str | Iterable[str]): Raw captions.

        Returns:
            str | Iterable[str]: Processed captions.
        """
        return text
    

class FormulaExtractionModel(HuggingFaceModel):
    """
    Specialized model for extracting mathematical formulas from images.
    """

    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained(FORMULA_EXTRACTION_MODEL_NAME)
        self.model = ORTModelForVision2Seq.from_pretrained(FORMULA_EXTRACTION_MODEL_NAME, use_cache=False)
        self.use_pixel_values = True
        self.prompt = ''
        self.config = {}

    def _postprocess(self, text: str | Iterable[str]):
        """
        Converts the raw output into LaTeX-formatted math expressions.

        Args:
            text (str | Iterable[str]): Raw output text from the model.

        Returns:
            str: LaTeX-formatted math expression.
        """
        return '$$' + text[0].replace(r'\\', '\\').replace(r'\tag', r'\quad \text') + '$$'