from mdify.src.utils import LAYOUT_DETECTION_MODEL_PATH, LAYOUT_DETECTION_CONFIG_FILE, IMAGES_SAVE_EXTENSION, digits_to_str
from mdify.src.models import PictureClassifier

from ultralytics import YOLO
from typing import Optional, Iterable
import supervision as sv
import os
import cv2
import logging
import yaml


class LayoutDetector:
    """
    Detects and organizes layout elements from a document page using a pre-trained YOLO model.
    This class is designed to process pages containing elements such as titles, headers, footnotes, and pictures, 
    allowing for efficient extraction, sorting, and saving of these components.

    Attributes:
        model: Pre-trained YOLO model for layout detection.
        config (dict): Configuration settings for the layout detection model.
        filename_separator (str): Separator used for naming cropped element files.
        picture_classifier (PictureClassifier): Classifier for recognizing picture types in the layout.
    """

    def __init__(self):

        self.model = YOLO(LAYOUT_DETECTION_MODEL_PATH)
        self.config = yaml.safe_load(LAYOUT_DETECTION_CONFIG_FILE.open('r'))
        self.filename_separator = '_'
        self.picture_classifier = PictureClassifier()
    
    def __get_num_of_columns(self, x_positions: Iterable[float], page_width: float, num_bins: Optional[int] = 10, density_threshold: Optional[float] = 0.5) -> int:
        """
        Estimates the number of columns in a document page based on the density of x-coordinates.

        Args:
            x_positions (Iterable[float]): X-coordinates of detected bounding boxes.
            page_width (float): Width of the page in pixels.
            num_bins (Optional[int]): Number of bins for dividing the page width. Default is 10.
            density_threshold (Optional[float]): Threshold for determining high-density regions. Default is 0.5.

        Returns:
            int: Number of columns detected in the page.
        """
        bin_width = page_width / num_bins
        density = [0] * num_bins

        for x in x_positions:
            bin_index = int(x // bin_width)
            density[bin_index] += 1

        # Normalize density values
        max_density = max(density)
        normalized_density = [d / max_density for d in density]

        # Count the number of high-density regions
        return sum(1 for d in normalized_density if d > density_threshold)
    
    def __custom_sort(self, elem):
        """
        Sorts layout elements based on their vertical and horizontal positions on the page.
        Elements are ordered first from top-to-bottom (y1) and then from left-to-right (x1).
        
        To support multicolumn formats, elements on the x-axis are assigned a higher weight the more they are on the right of the page.
        This way, elements at the bottom of the page can still be correctly positioned.
        
        Each element is also given a priority based on the type of element it is (e.g. footnotes must go at the bottom, headers at the top).
        
        Args:
            elem (tuple): A tuple containing a bounding box, label, and other metadata.

        Returns:
            tuple: Sorting key based on element type priority and position.
        """
        bbox, label = elem[0], elem[1]
        x1, y1, x2, y2 = bbox
        order_in_page = y1 + x1*self.n_cols

        if label == "Title":            
            return (0, order_in_page)   # First priority
        elif label == "Page-header":
            return (1, order_in_page)   # Second priority
        elif label == "Footnote":
            return (3, order_in_page)   # Second-last priority
        elif label == "Page-footer":
            return (4, order_in_page)   # Last priority
        else:
            return (2, order_in_page)   # Content

    def detect(self, image_path: str, page_nr: str, output_dir: Optional[str] = '', **kwargs):
        """
        Detects and extracts layout elements from a document page image.

        Args:
            image_path (str): Path to the input image of the document page.
            page_nr (str): Page number used for naming output files.
            output_dir (Optional[str]): Directory where extracted elements will be saved. Default is an empty string.
            **kwargs: Additional parameters for the YOLO model.

        Returns:
            List: A list of detected layout elements if `output_dir` is not provided. Otherwise, saves elements to the specified directory.
        """
        results = self.model(source=image_path, **(kwargs | self.config))[0]

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            return results

        elems = [(r.boxes.xyxy.numpy().flatten(), r.names[int(r.boxes.cls)], r.orig_img) for r in results]
        self.n_cols = self.__get_num_of_columns(x_positions=[e[0][0] for e in elems], page_width=max([e[0][2] for e in elems]))

        # Apply the custom sort
        sorted_elems = sorted(elems, key=self.__custom_sort)
        
        # Iterate through each element in elems and crop from the original image
        for i, (bbox, label, original_img) in enumerate(sorted_elems):
            x1, y1, x2, y2 = map(int, bbox)  # Convert bounding box to integers
            cropped_img = original_img[y1:y2, x1:x2]  # Crop using slicing
            
            if label == 'Picture':
                pred = self.picture_classifier.classify(cropped_img)
                label = self.picture_classifier.id2class[pred]

            # Save the cropped image
            output_path = os.path.join(output_dir, '.'.join([self.filename_separator.join([page_nr, digits_to_str(i+1), label]), IMAGES_SAVE_EXTENSION]))
            cv2.imwrite(output_path, cropped_img)

        logging.info(f"All elements have been cropped and saved in the '{output_dir}' folder.")

    def render(self, image_path: str, output_dir: str, **kwargs):
        """
        Renders annotated layout elements on the input document page image.

        Args:
            image_path (str): Path to the input image of the document page.
            output_dir (str): Directory where the annotated image will be saved.
            **kwargs: Additional parameters for the detection process.

        Saves:
            Annotated image with bounding boxes and labels drawn over the original image.
        """
        image = cv2.imread(image_path)
        
        results = self.detect(image_path, **kwargs)

        detections = sv.Detections.from_ultralytics(results)

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_dir, annotated_image)
        
