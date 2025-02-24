from mdify.src.utils import (
    clean_path, 
    get_filename, 
    digits_to_str, 
    get_file_extension, 
    open_image,
    IMAGES_SAVE_EXTENSION, 
    DOCUMENTS_SAVE_EXTENSION,
    ARTIFACTS_DEFAULT_DIR,
)
from mdify.src.layout import LayoutDetector
from mdify.src.extractors import ContentExtractor
from mdify.src.output import OutputWriter, OutputArtifact

from typing import Optional, Union
from pypdfium2 import PdfDocument
from glob import glob
import os
import io
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
        parse_directory(documents_dir: str, **kwargs): Parses multiple documents from a directory.
        parse(self, document: Union[str, bytes], document_name: str = None, document_type: str = None, **kwargs): Parses a single document and processes it.
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

    def parse_directory(self, documents_dir: str, **kwargs):
        """
        Parses multiple documents from a given directory.

        Args:
            documents_dir (str): Directory containing documents to be parsed.
            **kwargs: Additional arguments passed to the parse method.
        """
        for document in os.listdir(documents_dir):
            self.parse(os.path.join(documents_dir, document), **kwargs)

    def parse(self, document: Union[str, bytes], document_name: str = None, document_type: str = None, **kwargs):
        """
        Converts a document into Markdown.

        Args:
            document (Union[str, bytes]): Document to be parsed. If a string, it is the path to the document.
                                          If bytes, document_type and document_name must be specified.
            document_name (str, optional): Name of document if document is an instance of bytes. Defaults to None.
            document_type (str, optional): Type of document if document is an instance of bytes. Defaults to None.
            **kwargs: Additional keyword arguments passed to the rendering function and the extractor for each element.

        Raises:
            ValueError: If document_name and document_type are not provided when parsing bytes.
        """

        # Define document attributes
        if isinstance(document, str):
            self.document = clean_path(document)
            self.document_name = get_filename(document)
            self.doc_format = get_file_extension(self.document)
        elif isinstance(document, bytes):
            if document_name is None or document_type is None:
                raise ValueError("document_name and document_type must be provided when parsing bytes.")
            self.document = document
            self.document_name = get_filename(document_name)
            self.doc_format = document_type
        
        if self.doc_format in self.PIL_supported_formats:
            self.doc_format = 'image'

        self.base_output_folder = os.path.join(self.save_folder, self.document_name)
        self.pages_save_dir = os.path.join(self.base_output_folder, 'pages')
        self.layout_save_dir = os.path.join(self.base_output_folder, 'layout')
        self.artifacts_save_dir = os.path.join(self.base_output_folder, ARTIFACTS_DEFAULT_DIR)
        self.metadata_save_path = os.path.join(self.base_output_folder, self.metadata_filename)

        os.makedirs(self.pages_save_dir, exist_ok=True)
        os.makedirs(self.layout_save_dir, exist_ok=True)
        os.makedirs(self.artifacts_save_dir, exist_ok=True)

        # Define rendering kwargs
        rendering_kwarg_keys = ['scale']
        rendering_kwarg_keys = [key for key in rendering_kwarg_keys if key in kwargs]
        rendering_kwargs = {key: kwargs[key] for key in rendering_kwarg_keys}
        [kwargs.pop(k) for k in rendering_kwarg_keys]

        # Process document        
        try:
            eval(f"self._process_{self.doc_format}(**rendering_kwargs)")
        except Exception as e:
            logging.debug(e)
            logging.error(f'Format `{self.doc_format}` is not supported.')

        # Run OCR
        for image in os.listdir(self.pages_save_dir):
            page_nr = digits_to_str(get_filename(image))
            output_dir = os.path.join(self.layout_save_dir, page_nr)
            self.detector.detect(os.path.join(self.pages_save_dir, image), output_dir=output_dir, page_nr=page_nr)
        
        # Extract content
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

        # Write output
        OutputWriter.write(
            '\n\n'.join(texts),
            save_dir = self.base_output_folder,
            filename = self.document_name
        )
        
        # Clean up
        if not self.debug:
            shutil.rmtree(self.layout_save_dir)
            shutil.rmtree(self.pages_save_dir)

    def cleanup(self):
        """
        Deletes the directory where processed documents and artifacts are stored and all its contents.
        """
        shutil.rmtree(self.save_folder)
        
    def _process_image(self, **kwargs):
        """
        Saves the image in the appropriate folder.

        Args:
            **kwargs: Additional arguments. Currently not used.
        """
        page_nr = os.listdir(self.pages_save_dir)
        open_image(self.document).save(os.path.join(self.pages_save_dir, f'{page_nr+1}.{IMAGES_SAVE_EXTENSION}'))

    def _process_pdf(self, **kwargs):
        """
        Converts the pages of a PDF document into images to perform OCR.

        Args:
            **kwargs: Additional arguments passed to the rendering function.
        """
        document = self.document if isinstance(self.document, str) else io.BytesIO(self.document) 
        pdf = PdfDocument(document)
    
        if self.extract_metadata:
            json.dump(pdf.get_metadata_dict(), open(self.metadata_save_path, 'w'))

        for i, page in enumerate(pdf):
            bitmap = page.render(**kwargs)
            pil_image = bitmap.to_pil()
            pil_image.save(os.path.join(self.pages_save_dir, f'{i+1}.{IMAGES_SAVE_EXTENSION}'))

    @property
    def output_files_paths(self):
        """
        Returns:
            list: List of file paths to the files generated for each processed document.
        """
        if os.path.exists(self.save_folder):
            return glob(os.path.join(self.save_folder, f"**/*.{DOCUMENTS_SAVE_EXTENSION}"), recursive=True)
    
    @property
    def output_files(self):
        """       
        Returns:
            dict: A dictionary with the paths to the output files as keys and the corresponding file objects as values.
        """
        out_files = self.output_files_paths
        if out_files:
            files_content = [open(f, "r", encoding="utf-8").read() for f in out_files]
            return dict(zip(out_files, files_content))
    
    @property
    def metadata_paths(self):
        """
        Returns:
            dict: A dictionary with the paths to the output files as keys and the paths to the metadata files as values.
        """
        if os.path.exists(self.save_folder):
            return dict(zip(
                self.output_files_paths, 
                glob(os.path.join(self.save_folder, f"**/metadata.json"), recursive=True)
            ))

    @property
    def metadata(self):
        """
        Returns:
            dict: A dictionary where the keys are the paths to the output files and the values are the metadata objects associated with each document.
        """
        metadata_dict = self.metadata_paths
        if metadata_dict:
            return dict(zip(
                metadata_dict.keys(), 
                [json.load(open(metadata, 'r')) for metadata in metadata_dict.values()]
            ))