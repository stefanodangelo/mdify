from mdify.src.utils import (
    TABLE_OCR_CONFIG_PATH,
    TEXT_OCR_CONFIG_PATH,
    HEADER_OCR_CONFIG_PATH,
    IMAGES_SAVE_EXTENSION, 
    TABLES_SAVE_EXTENSION, 
    ARTIFACTS_DEFAULT_DIR, 
    get_filename, 
    clean_path, 
    xywh_to_xyxy, 
    SuppressOutput,
    open_image
)
from mdify.src.output import WriteMode, OutputArtifact
from mdify.src.models import ChartDeplotModel, ImageCaptioningModel, FormulaExtractionModel

from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr
from paddleocr import PaddleOCR, draw_ocr
import easyocr
from abc import abstractmethod
from typing import Iterable, Optional
from PIL import Image
import numpy as np
import pandas as pd 
import logging
import os
import yaml

# Suppress EasyOCR logs
logging.getLogger("easyocr").setLevel(logging.ERROR)

class OCR:    
    """
    Abstract base class for OCR processing. It provides a common interface and workflow 
    for derived OCR classes.

    Methods:
        process: Main method to process an image and perform OCR.
        process_results: Abstract method to be implemented by subclasses for processing OCR results.
    """

    def process(
        self,
        image_path: str, 
        save_dir: str, 
        filename: str, 
        save_artifacts: Optional[OutputArtifact] = OutputArtifact.NONE,
        write_mode: Optional[WriteMode] = WriteMode.EMBEDDED,
        debug: Optional[bool] = False,
        **kwargs
    ):
        """
        Processes the image and initializes OCR attributes.

        Args:
            image_path (str): Path to the input image.
            save_dir (str): Directory to save OCR artifacts.
            filename (str): Base filename for saving artifacts.
            save_artifacts (Optional[OutputArtifact]): Specifies which artifacts to save.
            write_mode (Optional[WriteMode]): Determines the output format of extracted text.
            debug (Optional[bool]): If True, saves a debug image.
            **kwargs: Additional arguments for customization.

        Raises:
            Exception: If the image cannot be processed.
        """

        self.write_mode = write_mode

        self.image_path = image_path
        self.image = open_image(image_path, to_numpy=True)
        self.filename = get_filename(filename)

        self.save_artifacts = save_artifacts

        if debug:
            debug_save_path = os.path.join(save_dir, f"{self.filename}_debug.{IMAGES_SAVE_EXTENSION}")
            Image.fromarray(self.image).save(debug_save_path)
        
        self.extracted_text = ''
        self.ocr_error_msg = r"OCR can't be done on the image."

        self.process_results(save_dir=save_dir, save_artifacts=save_artifacts, **kwargs)

    @abstractmethod
    def process_results(self, **kwargs):
        """
        Abstract method to process OCR results. Subclasses must implement this method.
        """
        pass


class TextRecognizer(OCR):
    """
    Class for recognizing text from images using OCR.
    Two different models are used for headers and paragraphs as it has been observed that using only one of them for both yields less accurate results.

    Attributes:
        paragraph_model_config (dict): Configuration for paragraph-level OCR.
        header_reader (easyocr.Reader): Reader instance for header-level OCR.
        det_processor, det_model, rec_model, rec_processor: Components for OCR processing.
        default_element_type (str): Default type of element to process ('paragraph').
    """

    def __init__(self):
        with SuppressOutput():
            self.paragraph_model_config = yaml.safe_load(open(TEXT_OCR_CONFIG_PATH, 'r'))
            self.header_reader = easyocr.Reader(**yaml.safe_load(open(HEADER_OCR_CONFIG_PATH, 'r')))
            self.det_processor, self.det_model, self.rec_model, self.rec_processor = load_det_processor(), load_det_model(), load_rec_model(), load_rec_processor()
            self.default_element_type = 'paragraph'

    def _process_paragraph(self):
        """
        Processes the image to extract paragraph-level text.
        """
        with SuppressOutput():
            predictions = run_ocr(
                [open_image(self.image, to_numpy=False)], 
                det_model=self.det_model, 
                det_processor=self.det_processor, 
                rec_model=self.rec_model, 
                rec_processor=self.rec_processor, 
                **self.paragraph_model_config
            )
            lines = predictions[0].text_lines
            self.texts = [line.text + ('\\' if i < len(lines) - 1 else '') for i, line in enumerate(lines)] # add explicit line break except for last line

    def _process_header(self):
        """
        Processes the image to extract header-level text.
        """
        results = self.header_reader.readtext(self.image)
        self.texts = [text for _, text, _ in results]

    def process_results(self, **kwargs):
        """
        Processes OCR results based on the specified element type (paragraph or header).

        Args:
            element_to_process (str): either 'header' or 'paragraph'.
        """
        elem_type = kwargs.get('element_to_process', self.default_element_type)
        elem_type = elem_type if elem_type in ['header', self.default_element_type] else self.default_element_type

        try:
            eval(f'self._process_{elem_type}()')
            self.extracted_text = '\n'.join(self.texts)
        except Exception as e:
            print(e)
            logging.warning(self.ocr_error_msg)


class TableRecognizer(OCR):
    """
    Class for recognizing and organizing table data from images.

    Attributes:
        config (dict): Configuration for table OCR model.
        model (PaddleOCR): Instance of PaddleOCR model.
    """

    def __init__(self):
        self.config = yaml.safe_load(open(TABLE_OCR_CONFIG_PATH, 'r'))
        self.model = PaddleOCR(**self.config, show_log=False)

    def ocr(self, img: np.ndarray, cls: bool = True, **kwargs):
        """
        Performs OCR on the provided image and extracts table text and bounding boxes.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            cls (bool): Whether to use classification.
            **kwargs: Additional arguments passed to the OCR model.
        """
        self.bboxes = []
        self.texts = []
        self.scores = []
        self.result = self.model.ocr(img, cls=cls, **kwargs)

        for res in self.result:
            for line in res:
                text, score = line[1]
                self.texts.append(text)
                self.bboxes.append(line[0])
                self.scores.append(score)
    
    def render(self, img: np.ndarray | str, save_full_path: str, font_path: str = '../utils/simfang.ttf', **kwargs):
        """
        Renders the recognized data onto the input image and saves it.

        Args:
            img (np.ndarray | str): Input image as an array or path.
            save_full_path (str): Full path to save the rendered image.
            font_path (str): Path to the font used for rendering.
            **kwargs: Additional arguments for rendering.
        """
        img = open_image(img, to_numpy=True)
        self.model.ocr(img, **kwargs)
        im_show = draw_ocr(self.result, self.bboxes, self.texts, self.scores, font_path=font_path)
        Image.fromarray(im_show).save(os.path.join(clean_path(save_full_path), '.'.join([get_filename(save_full_path), IMAGES_SAVE_EXTENSION])))


    def __find_column_boundaries(self, bboxes: Iterable[Iterable[float]], tolerance: int = 10) -> Iterable[float]:
        """
        Finds boundaries of table columns based on bounding boxes.

        Args:
            bboxes (Iterable[Iterable[float]]): List of bounding boxes.
            tolerance (int): Margin of error in pixels for boundary grouping.

        Returns:
            Iterable[float]: Sorted list of column boundaries.
        """
        # Determine how many columns the table has
        boundaries = set()
        similar_boundary_exists = False
        for bbox in bboxes:
            x_min = bbox[0]
            for boundary in boundaries:
                similar_boundary_exists = abs(boundary - x_min) <= tolerance
                if similar_boundary_exists:
                    break # exit when similar boundary is found
            if not similar_boundary_exists:
                boundaries.add(x_min)
        
        return sorted(list(boundaries))
    

    def __find_header(self, column_boundaries: Iterable[float], bboxes: np.ndarray, tolerance: int = 10) -> int:
        """
        Identifies the row index where the table header ends.

        Args:
            column_boundaries (Iterable[float]): List of column boundaries.
            bboxes (np.ndarray): Array of bounding boxes.
            tolerance (int): Margin of error in pixels for header detection.

        Returns:
            int: Row index where the header ends.
        """
        header_end = 1 # header is assumed to be one row
        # TODO: REFINE HEADER DETECTION ALGORITHM
        return header_end
    

    def process_results(self, **kwargs):
        """
        Organizes text into table columns based on bounding box x-coordinates and returns a DataFrame.
        Ensures every column is populated with either text or None for all rows.

        Args:
            row_tolerance (int): Margin of error in pixels to assign elements to the correct row.
            col_tolerance (int): Margin of error in pixels to assign elements to the correct column.
        """

        row_tolerance = kwargs.get('row_tolerance', 20)
        col_tolerance = kwargs.get('col_tolerance', 50)

        ## Perform OCR
        try:
            self.ocr(self.image)
        except:
            logging.warning(self.ocr_error_msg)
            return

        ocr_data = dict(zip([xywh_to_xyxy(b) for b in self.bboxes], self.texts))

        ## Save results
        # Find column boundaries and value at which header ends
        column_boundaries = self.__find_column_boundaries(ocr_data.keys(), tolerance=col_tolerance)
        header_end = self.__find_header(column_boundaries, ocr_data.keys(), tolerance=col_tolerance)

        # Initialize a list to hold rows of text, one row per y-coordinate range
        rows = []

        # Loop through bounding boxes and group text by y-coordinate proximity (row structure)
        for bbox, text in ocr_data.items():
            x_min, y_min, x_max, y_max = bbox
            row_found = False
            
            # Attempt to place the current text in an existing row
            for row in rows:
                if abs(row['y_min'] - y_min) < row_tolerance:  # Adjust threshold as needed for row grouping
                    row['entries'].append((x_min, text))
                    row_found = True
                    break

            # If no matching row found, create a new one
            if not row_found:
                rows.append({'y_min': y_min, 'entries': [(x_min, text)]})

        # Initialize the final structured table as a list of lists
        structured_table = []

        for row in rows:
            # Sort entries in the row by x_min to align text with columns
            row['entries'].sort(key=lambda x: x[0])

            # Start with a row filled with None for every column
            structured_row = [None] * len(column_boundaries)

            # Place text in the appropriate column based on x_min
            for x_min, text in row['entries']:
                for col_index in range(len(column_boundaries)):
                    if column_boundaries[col_index] - col_tolerance <= x_min <= column_boundaries[col_index] + col_tolerance:
                        structured_row[col_index] = text
                        break

            # Add the structured row to the table
            structured_table.append(structured_row)

        # Create a DataFrame from the structured table
        df = pd.DataFrame(structured_table)

        # Rename columns to represent their index in the table
        df.columns = [f"Column {i}" for i in range(len(df.columns))]

        # Identify row number at which data starts (and header ends) ignoring nulls as they are not identified by the OCR algorithm
        # data_row_index = df[df['Column 0'] == [v for v in df.values.flatten() if v is not None][header_end]].index[0]
        data_row_index = df.iloc[:header_end].reset_index(drop=True).index[0]

        if data_row_index > 0:
            # Apply the header hierarchy to the DataFrame
            df.columns = pd.MultiIndex.from_arrays(df.iloc[:data_row_index].values)
            df = df.iloc[data_row_index:]

            # Create a copy of the DataFrame with a Multi-Level Header
            new_df = pd.DataFrame(df.values, columns=pd.MultiIndex.from_tuples(df.columns))

            # Flatten Headers
            flattened_headers = [" - ".join(
                filter(None, # filters anything "falsy", e.g. None, 0, etc.
                    map(
                        str, # convert every column to a string
                        filter(lambda x: isinstance(x, str), col) # only keeps string columns, i.e. discards nulls
                    )
                )).strip() 
                for col in new_df.columns.to_flat_index()
            ]
            new_df.columns = flattened_headers

            df = new_df.copy()

        if self.write_mode in [WriteMode.EMBEDDED, WriteMode.DESCRIBED]:
            self.extracted_text = df.to_markdown(index=False)
        elif self.write_mode == WriteMode.PLACEHOLDER:
            self.extracted_text = f'<-- table ({self.filename}) -->'
        
        save_path = os.path.join(kwargs.get('save_dir', ARTIFACTS_DEFAULT_DIR), '.'.join([self.filename, TABLES_SAVE_EXTENSION]))
        
        if self.save_artifacts.value in [
            OutputArtifact.ALL.value, 
            OutputArtifact.ONLY_TABLES.value, 
            OutputArtifact.PICTURES_AND_TABLES.value, 
            OutputArtifact.PICTURES_TABLES_AND_CHARTS.value, 
            OutputArtifact.PICTURES_TABLES_AND_FORMULAS.value,
            OutputArtifact.TABLES_CHARTS_AND_FORMULAS.value,
            OutputArtifact.TABLES_AND_FORMULAS.value,
            OutputArtifact.TABLES_AND_CHARTS.value
        ]:
            eval(f'df.to_{TABLES_SAVE_EXTENSION}(save_path, header=False)')

class PictureRecognizer(OCR):
    """
    Class for recognizing and processing image-based data such as charts, captions, or formulas.

    Attributes:
        chart_qa_model: Model for extracting data from charts.
        captioning_model: Model for generating captions for images.
        formula_extraction_model: Model for extracting mathematical formulas.
    """
    def __init__(self):
        self.chart_qa_model = ChartDeplotModel()
        self.captioning_model = ImageCaptioningModel()
        self.formula_extraction_model = FormulaExtractionModel()

    def __make_save_path(self, **kwargs):
        """
        Creates the save path for the image artifact.
        """
        self.save_path = os.path.join(kwargs.get('save_dir', ARTIFACTS_DEFAULT_DIR), '.'.join([self.filename, IMAGES_SAVE_EXTENSION]))

    def __save(self, **kwargs):
        """
        Saves the processed image.
        """
        self.__make_save_path(extension=IMAGES_SAVE_EXTENSION, **kwargs)
        img = self.image if type(self.image) == Image else Image.fromarray(self.image)
        img.save(self.save_path)

    def process_results(self, **kwargs):
        """
        Processes the image results based on its type (chart, formula, or general image).

        Args:
            save_dir (str): Directory to save OCR artifacts.
        """
        self.__make_save_path(**kwargs)
        is_chart = 'chart' in self.filename.lower()
        is_formula = 'formula' in self.filename.lower()

        if self.write_mode == WriteMode.EMBEDDED:
            self.extracted_text = f"![{self.filename}]({self.save_path})"
            self.__save(**kwargs)
            return
        elif self.write_mode == WriteMode.PLACEHOLDER:
            self.extracted_text = f'<-- image ({self.filename}) -->'
        elif self.write_mode == WriteMode.DESCRIBED:
            if is_chart:
                result = self.chart_qa_model.predict(self.image_path)
                self.extracted_text = result.to_markdown(index=False)
            elif is_formula:
                self.extracted_text = self.formula_extraction_model.predict(self.image_path)
            else:
                self.extracted_text = self.captioning_model.predict(self.image_path)

        if is_chart and self.save_artifacts.value in [
            OutputArtifact.ALL.value, 
            OutputArtifact.ONLY_CHARTS.value, 
            OutputArtifact.CHARTS_AND_FORMULAS.value, 
            OutputArtifact.PICTURES_AND_CHARTS.value,
            OutputArtifact.TABLES_AND_CHARTS.value, 
            OutputArtifact.TABLES_CHARTS_AND_FORMULAS.value,
            OutputArtifact.PICTURES_CHARTS_AND_FORMULAS.value,
            OutputArtifact.PICTURES_TABLES_AND_CHARTS.value
        ]:
            eval(f'result.to_{TABLES_SAVE_EXTENSION}(self.save_path, header=False)')
        elif is_formula and self.save_artifacts.value in [
            OutputArtifact.ALL.value, 
            OutputArtifact.ONLY_FORMULAS.value, 
            OutputArtifact.CHARTS_AND_FORMULAS.value, 
            OutputArtifact.TABLES_AND_FORMULAS.value,
            OutputArtifact.PICTURES_AND_FORMULAS.value, 
            OutputArtifact.TABLES_CHARTS_AND_FORMULAS.value,
            OutputArtifact.PICTURES_CHARTS_AND_FORMULAS.value,
            OutputArtifact.PICTURES_TABLES_AND_FORMULAS.value
        ]:
            self.__save(**kwargs)
        elif self.save_artifacts.value in [
            OutputArtifact.ALL.value, 
            OutputArtifact.ONLY_PICTURES.value, 
            OutputArtifact.PICTURES_AND_CHARTS.value, 
            OutputArtifact.PICTURES_AND_TABLES.value,
            OutputArtifact.PICTURES_AND_FORMULAS.value, 
            OutputArtifact.PICTURES_TABLES_AND_FORMULAS.value,
            OutputArtifact.PICTURES_CHARTS_AND_FORMULAS.value,
            OutputArtifact.PICTURES_TABLES_AND_CHARTS.value
        ]:
            self.__save(**kwargs)