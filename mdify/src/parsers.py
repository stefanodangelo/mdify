from mdify.src.utils import (
    clean_path, 
    get_filename, 
    digits_to_str, 
    get_file_extension, 
    open_image,
    IMAGES_SAVE_EXTENSION, 
    ARTIFACTS_DEFAULT_DIR
)
from mdify.src.layout import LayoutDetector
from mdify.src.extractors import ContentExtractor
from mdify.src.output import OutputWriter, OutputArtifact

from typing import Optional
from pypdfium2 import PdfDocument
import os
import json
import shutil
import logging


class DocumentParser:
    """
    A class that processes documents (PDFs or images) to extract their content and save artifacts like images, tables, charts, and formulas.
    
    Attributes:
        save_folder (str): Folder where the processed documents will be saved.
        extract_metadata (bool): Flag to indicate if metadata should be extracted.
        metadata_filename (str): The filename for saving metadata.
        save_artifacts (OutputArtifact): Specifies the types of artifacts to save.
        debug (bool): Flag to control debug mode. If `True`, it will keep temporary folders containing all pages and all elements extracted in each page in image format. 
        PIL_supported_formats (list): List of supported image formats by PIL.
        detector (LayoutDetector): Layout detection object.
        extractor (ContentExtractor): Content extraction object.
    
    Methods:
        parse_multiple(documents_dir: str, **kwargs): Parses multiple documents from a directory.
        parse(document_path: str, **kwargs): Parses a single document and processes it.
    """

    def __init__(
            self, 
            save_folder: Optional[str] = 'processed', 
            extract_metadata: Optional[bool] = True, 
            save_artifacts: Optional[OutputArtifact] = OutputArtifact.NONE,
            debug: Optional[bool] = False
        ):
        self.save_folder = clean_path(save_folder)
        self.extract_metadata = extract_metadata
        self.metadata_filename = 'metadata.json'
        self.save_artifacts = save_artifacts
        self.debug = debug
        self.PIL_supported_formats = [
            "bmp", "dib", "eps", "gif", "icns", "ico", "im", "jpeg", "jpg",
            "msp", "pcx", "png", "ppm", "sgi", "spi", "tiff", "webp", "xbm", "tga"
        ]

        self.detector = LayoutDetector()
        self.extractor = ContentExtractor() 

    def parse_multiple(self, documents_dir: str, **kwargs):
        """
        Parses multiple documents from a given directory.

        Args:
            documents_dir (str): Directory containing documents to be parsed.
            **kwargs: Additional arguments passed to the parse method.
        """
        for document in os.listdir(documents_dir):
            self.parse(os.path.join(documents_dir, document), **kwargs)

    def parse(self, document_path: str, **kwargs):    
        """
        Parses a single document, processes its pages, extracts content, and saves the output.

        Args:
            document_path (str): Path to the document to be parsed.
            **kwargs: Additional arguments passed to the content extraction methods.
        """
        rendering_kwarg_keys = ['scale']
        rendering_kwargs = {key: kwargs[key] for key in rendering_kwarg_keys if key in kwargs}
        [kwargs.pop(k) for k in rendering_kwarg_keys]

        self.document_path = clean_path(document_path)
        self.document_name = get_filename(document_path)
        self.base_output_folder = os.path.join(self.save_folder, self.document_name)
        self.pages_save_dir = os.path.join(self.base_output_folder, 'pages')
        self.layout_save_dir = os.path.join(self.base_output_folder, 'layout')
        self.artifacts_save_dir = os.path.join(self.base_output_folder, ARTIFACTS_DEFAULT_DIR)
        self.metadata_save_path = os.path.join(self.base_output_folder, self.metadata_filename)

        os.makedirs(self.pages_save_dir, exist_ok=True)
        os.makedirs(self.layout_save_dir, exist_ok=True)
        os.makedirs(self.artifacts_save_dir, exist_ok=True)

        doc_format = get_file_extension(self.document_path)
        if doc_format in self.PIL_supported_formats:
            doc_format = 'image'

        try:
            eval(f"self._process_{doc_format}(**rendering_kwargs)")
        except Exception as e:
            print(e)
            logging.error(f'Format `{doc_format}` is not supported.')

        for image in os.listdir(self.pages_save_dir):
            page_nr = digits_to_str(get_filename(image))
            output_dir = os.path.join(self.layout_save_dir, page_nr)
            self.detector.detect(os.path.join(self.pages_save_dir, image), output_dir=output_dir, page_nr=page_nr)
        
        texts = []
        for page in os.listdir(self.layout_save_dir):
            for image in os.listdir(os.path.join(self.layout_save_dir, page)):
                filename = get_filename(image)
                img_path = os.path.join(self.layout_save_dir, page, image)

                extract_type = (
                    filename
                    .split(self.detector.filename_separator)[-1]
                    .replace("-", "_")
                    .replace('chart', 'picture')
                    .replace('formula', 'picture')
                    .lower()
                )
                result = self.extractor.extract(
                    image_path=img_path,
                    extract_type=extract_type,
                    save_dir=self.artifacts_save_dir,
                    filename=filename,
                    save_artifacts=self.save_artifacts,
                    **kwargs
                )

                texts.append(result)

        OutputWriter.write(
            '\n\n'.join(texts),
            save_dir = self.base_output_folder,
            filename = self.document_name
        )
        
        if not self.debug:
            shutil.rmtree(self.layout_save_dir)
            shutil.rmtree(self.pages_save_dir)

    def _process_image(self, **kwargs):
        """
        Saves the image in the appropriate folder.

        Args:
            **kwargs: Additional arguments. Currently not used.
        """
        page_nr = os.listdir(self.pages_save_dir)
        open_image(self.document_path).save(os.path.join(self.pages_save_dir, f'{page_nr+1}.{IMAGES_SAVE_EXTENSION}'))

    def _process_pdf(self, **kwargs):
        """
        Converts the pages of a PDF document into images to perform OCR.

        Args:
            **kwargs: Additional arguments passed to the rendering function.
        """
        pdf = PdfDocument(self.document_path)
    
        if self.extract_metadata:
            json.dump(pdf.get_metadata_dict(), open(self.metadata_save_path, 'w'))

        for i, page in enumerate(pdf):
            bitmap = page.render(**kwargs)
            pil_image = bitmap.to_pil()
            pil_image.save(os.path.join(self.pages_save_dir, f'{i+1}.{IMAGES_SAVE_EXTENSION}'))