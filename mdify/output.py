from enum import Enum
import os

class WriteMode(Enum):
    """
    Enum representing the modes for writing content with embedded or referenced artifacts.

    Modes:
        EMBEDDED (int): Artifacts (e.g., images, tables) are directly embedded in the output content.
        PLACEHOLDER (int): Placeholders for artifacts are added in the output content, requiring external references.
        DESCRIBED (int): Artifacts are described textually, without direct embedding or placeholders.
    """
    EMBEDDED = 0
    PLACEHOLDER = 1
    DESCRIBED = 2


class OutputArtifact(Enum):
    """
    Enum representing the types of artifacts to be saved.

    Types:
        NONE (int): No artifacts are saved.
        ONLY_PICTURES (int): Only images are saved.
        ONLY_TABLES (int): Only tables are saved.
        ONLY_CHARTS (int): Only charts are saved.
        ONLY_FORMULAS (int): Only formulas are saved.
        PICTURES_AND_TABLES (int): Both images and tables are saved.
        PICTURES_AND_CHARTS (int): Both images and charts are saved.
        PICTURES_AND_FORMULAS (int): Both images and formulas are saved.
        TABLES_AND_CHARTS (int): Both tables and charts are saved.
        TABLES_AND_FORMULAS (int): Both tables and formulas are saved.
        CHARTS_AND_FORMULAS (int): Both charts and formulas are saved.
        PICTURES_TABLES_AND_CHARTS (int): Images, tables, and charts are saved.
        PICTURES_TABLES_AND_FORMULAS (int): Images, tables, and formulas are saved.
        PICTURES_CHARTS_AND_FORMULAS (int): Images, charts, and formulas are saved.
        TABLES_CHARTS_AND_FORMULAS (int): Tables, charts, and formulas are saved.
        ALL (int): All artifacts (images, tables, charts, formulas) are saved.
    """
    NONE = 0
    ONLY_PICTURES = 1
    ONLY_TABLES = 2
    ONLY_CHARTS = 3
    ONLY_FORMULAS = 4
    PICTURES_AND_TABLES = 5
    PICTURES_AND_CHARTS = 6
    PICTURES_AND_FORMULAS = 7
    TABLES_AND_CHARTS = 8
    TABLES_AND_FORMULAS = 9
    CHARTS_AND_FORMULAS = 10
    PICTURES_TABLES_AND_CHARTS = 11
    PICTURES_TABLES_AND_FORMULAS = 12
    PICTURES_CHARTS_AND_FORMULAS = 13
    TABLES_CHARTS_AND_FORMULAS = 14
    ALL = 15


class OutputWriter:
    """
    Handles the process of writing content and optionally saving specified types of artifacts.

    Methods:
        write(content: str, save_dir: str, filename: str):
            Writes the content to a file in the specified directory, creating the directory if needed.
    """

    @staticmethod
    def write(content: str, save_dir: str, filename: str):
        """
        Writes the content to a markdown file in the specified directory.

        Args:
            content (str): The content to write to the file.
            save_dir (str): Directory where the file will be saved.
            filename (str): Name of the file (without extension).
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '.'.join([filename, 'md']))
        with open(save_path, 'w') as f:
            f.write(content)
