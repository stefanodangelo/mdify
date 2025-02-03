from huggingface_hub import snapshot_download
from typing import Optional, Union
from PIL import Image
from io import BytesIO
import numpy as np
import sys
import os

# Global Variables
"""
Global constants used throughout the script for configuring paths, file extensions, and model settings.
"""

# Path and Separator Constants
FOLDERS_SEPARATOR = '/'  # Defines the folder separator for paths.
IMAGES_SAVE_EXTENSION = 'jpg'  # Default file extension for saving images.
TABLES_SAVE_EXTENSION = 'csv'  # Default file extension for saving tables.
PDF_EXTENSION = 'pdf'  # Default file extension for saving PDFs.

# Model and Configuration Paths
LAYOUT_DETECTION_MODEL_PATH = snapshot_download(repo_id='omoured/YOLOv10-Document-Layout-Analysis', allow_patterns='*l_best.pt')  # Path to the layout detection model.
LAYOUT_DETECTION_MODEL_PATH = os.path.join(LAYOUT_DETECTION_MODEL_PATH, os.listdir(LAYOUT_DETECTION_MODEL_PATH)[0])
LAYOUT_DETECTION_CONFIG_PATH = 'mdify/configs/layout_detection_config.yml'  # Path to the layout detection config file.
TABLE_OCR_CONFIG_PATH = 'mdify/configs/paddle_ocr_config.yml'  # Path to the OCR configuration for tables.
TEXT_OCR_CONFIG_PATH = 'mdify/configs/surya_ocr_config.yml'  # Path to the OCR configuration for text.
HEADER_OCR_CONFIG_PATH = 'mdify/configs/easy_ocr_config.yml'  # Path to the OCR configuration for headers.
CHART_DETECTION_MODEL_PATH = snapshot_download(repo_id='stefanodangelo/chartdet', allow_patterns='*.pt')  # Path to the chart detection model.
CHART_DETECTION_MODEL_PATH = os.path.join(CHART_DETECTION_MODEL_PATH, os.listdir(CHART_DETECTION_MODEL_PATH)[0])
CHART_DETECTION_MODEL_NAME = 'microsoft/swin-large-patch4-window7-224'  # Model name for chart detection.
CHART_DETECTION_N_CLASSES = 2  # Number of classes in the chart detection model.

# Artifact Directory
ARTIFACTS_DEFAULT_DIR = 'artifacts'  # Default directory for saving artifacts.

# Chart VQA Model Configurations
CHART_VQA_MODEL_NAME_CLASS = ('google/deplot', 'Pix2Struct')  # Model name and class for chart VQA.
CHART_VQA_PROMPT = 'Generate the underlying data table of this figure.'  # Prompt for the chart VQA model.
CHART_VQA_MODEL_CONFIG_PATH = 'mdify/configs/google_vqa_config.yml'  # Path to the chart VQA config file.

# Image VQA Model Configurations
IMAGE_VQA_MODEL_NAME_CLASS = ('Salesforce/blip-image-captioning-large', 'Blip')  # Model name and class for image VQA.
IMAGE_VQA_MODEL_CONFIG_PATH = 'mdify/configs/google_vqa_config.yml'  # Path to the image VQA config file.

# Formula Extraction Model
FORMULA_EXTRACTION_MODEL_NAME = 'breezedeus/pix2text-mfr'  # Model name for formula extraction.

# Utility Functions
clean_path = lambda path: path.replace('\\', FOLDERS_SEPARATOR)

xywh_to_xyxy = lambda bbox: tuple(np.array(bbox).min(axis=0)) + tuple(np.array(bbox).max(axis=0))

digits_to_str = lambda x: '0' + str(x) if str(x).isnumeric() and abs(int(x)) < 10 else str(x)

get_file_extension = lambda filename: filename[filename.rindex('.')+1:]

def get_filename(path: str, include_extension: Optional[bool] = False) -> str:
    """
    Extracts the filename from a given path.

    Args:
        path (str): The full path of the file.
        include_extension (bool, optional): Whether to include the file extension. Defaults to False.
    Returns:
        str: The extracted filename.
    """
    path = clean_path(path)
    # filename = path.split(FOLDERS_SEPARATOR)[-1] if FOLDERS_SEPARATOR in path else path
    filename = os.path.basename(path)
    return filename if include_extension or '.' not in filename else filename[:filename.rindex('.')]

def convert_to_jpeg(im: Image) -> Image:
    """
    Converts an image to JPEG format.

    Args:
        im (PIL.Image): The input image.
    Returns:
        PIL.Image: The converted image in JPEG format.
    """
    with BytesIO() as f:
        im.convert('RGB').save(f, format='JPEG')
        f.seek(0)
        img = Image.open(f)
        img.load()
        return img

def open_image(img: Union[np.ndarray, str], to_numpy: bool = False) -> Union[np.ndarray, Image]:
    """
    Opens an image file or array, converting it to JPEG format if necessary.

    Args:
        img (str | np.ndarray): Path to the image file or an image array.
        to_numpy (bool, optional): Whether to return the image as a NumPy array. Defaults to False.
    Returns:
        Union[np.ndarray, PIL.Image]: The opened image.
    """
    img = convert_to_jpeg(Image.open(img)) if isinstance(img, str) else Image.fromarray(img)
    return np.array(img) if to_numpy else img

def convert_image_to_pdf(image_path: str, pdf_path: str):
    """
    Converts an image file to a PDF.

    Args:
        image_path (str): Path to the input image file.
        pdf_path (str): Path to save the output PDF file.
    """
    image = Image.open(image_path)
    image = image.convert('RGB')
    image.save(pdf_path, "PDF")


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr