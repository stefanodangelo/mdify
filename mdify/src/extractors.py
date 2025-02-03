from mdify.src.ocr import TextRecognizer, TableRecognizer, PictureRecognizer

from typing import Optional

class ContentExtractor:
    """
    Extracts content from a document image based on specified types such as text, tables, pictures, and more.
    This class delegates extraction tasks to specialized recognizers for text, tables, and pictures.

    Attributes:
        text_recognizer (TextRecognizer): Recognizer for extracting textual elements.
        table_recognizer (TableRecognizer): Recognizer for extracting table elements.
        picture_recognizer (PictureRecognizer): Recognizer for extracting picture elements.
    """

    def __init__(self):
        self.text_recognizer = TextRecognizer()
        self.table_recognizer = TableRecognizer()
        self.picture_recognizer = PictureRecognizer()

    def extract(self, image_path: str, extract_type: str, **kwargs) -> str:
        """
        Method factory that extracts content from the specified image based on the type of element requested.

        Args:
            image_path (str): Path to the input image of the document.
            extract_type (str): Type of content to extract (e.g., 'text', 'table', 'picture', etc.).
            **kwargs: Additional parameters for the specific extraction method.

        Returns:
            str: Extracted content as a string.
        """
        return eval(f"self._extract_{extract_type}(image_path, **kwargs)")
    
    def _extract_table(self, image_path: str, **kwargs):
        """
        Extracts table content from the image using the table recognizer.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for the table recognizer.

        Returns:
            str: Extracted table content as a string.
        """
        self.table_recognizer.process(image_path, **kwargs)
        return self.table_recognizer.extracted_text

    def _extract_text(self, image_path: str, element_type: Optional[str] = None, **kwargs):
        """
        Extracts text content from the image using the text recognizer.

        Args:
            image_path (str): Path to the input image.
            element_type (Optional[str]): Specific type of text element to extract (e.g., 'paragraph', 'header').
                                          Default is 'paragraph'.
            **kwargs: Additional parameters for the text recognizer.

        Returns:
            str: Extracted text content as a string.
        """
        self.text_recognizer.process(image_path, element_to_process=element_type or 'paragraph', **kwargs)
        return self.text_recognizer.extracted_text

    def _extract_picture(self, image_path: str, **kwargs):
        """
        Extracts picture-related content from the image using the picture recognizer.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for the picture recognizer.

        Returns:
            str: Extracted picture content as a string.
        """
        self.picture_recognizer.process(image_path, **kwargs)
        return self.picture_recognizer.extracted_text

    def _extract_chart(self, image_path: str, **kwargs):
        """
        Extracts chart-related content by delegating to the picture recognizer.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for the picture recognizer.

        Returns:
            str: Extracted chart content as a string.
        """
        return self._extract_picture(image_path, **kwargs)

    def _extract_list_item(self, image_path: str, **kwargs):
        """
        Extracts a list item by prefixing extracted text with a bullet point.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for text extraction.

        Returns:
            str: Extracted list item content as a string.
        """
        return '\t- ' + self._extract_text(image_path, **kwargs)

    def _extract_page_header(self, image_path: str, **kwargs):
        """
        Extracts the page header from the image.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for text extraction.

        Returns:
            str: Extracted page header content as a string.
        """
        return self._extract_text(image_path, element_type='header', **kwargs)

    def _extract_title(self, image_path: str, **kwargs):
        """
        Extracts the title from the image, formatted as a Markdown header.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for text extraction.

        Returns:
            str: Extracted title content as a string.
        """
        return '# ' + self._extract_text(image_path, element_type='header', **kwargs)

    def _extract_section_header(self, image_path: str, **kwargs):
        """
        Extracts a section header, formatted as a secondary Markdown header.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for text extraction.

        Returns:
            str: Extracted section header content as a string.
        """
        return '#' + self._extract_title(image_path, **kwargs)

    def _extract_caption(self, image_path: str, **kwargs):
        """
        Extracts a caption, prefixed with the word 'Caption:'.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for text extraction.

        Returns:
            str: Extracted caption content as a string.
        """
        return 'Caption: ' + self._extract_text(image_path, **kwargs)

    def _extract_footnote(self, image_path: str, **kwargs):
        """
        Extracts a footnote from the image.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for text extraction.

        Returns:
            str: Extracted footnote content as a string.
        """
        return self._extract_text(image_path, **kwargs)

    def _extract_page_footer(self, image_path: str, **kwargs):
        """
        Extracts the page footer from the image.

        Args:
            image_path (str): Path to the input image.
            **kwargs: Additional parameters for text extraction.

        Returns:
            str: Extracted page footer content as a string.
        """
        return self._extract_text(image_path, **kwargs)