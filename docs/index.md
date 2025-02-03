# MDify

Welcome to the official documentation of **MDify** 🤗

## Contents
1. [Reference](reference.md): contains documentation and references to the source code.
2. [Usage](usage.md): contains examples of how to use the library.

## 1. Introduction
**MDify** is a powerful Python library for converting PDFs (or image-based documents) into clean, structured Markdown.

## 2. Features
✔️ **Handles complex layouts** - Extracts text, tables, and visual elements with precision

🖼️ **Preserves images & charts** - Gives the option to save and reuse extracted visuals for Computer Vision tasks

🎯 **Optimized for accuracy** - Combines layout detection and OCR to extract text from documents

🤖 **Preprocessing for LLM applications** - Converts documents to Markdown, which is popular for LLM training and fine-tuning tasks

🛠️ **Debug mode** - Save intermediate document elements as images for analysis

**Notes**:
- The first run will take ~2 minutes to download the necessary models.
- Currently, MDify can only handle PDFs and images (such as text extracts, document scans, etc.). Therefore, please make sure that your documents meet the requirements in order for the tool to work best.
- Diagrams are not supported yet, therefore if you use the `DESCRIBED` write mode they may be analyzed incorrectly.

## 3. Contributing
MDify is an independent, open-source project developed and maintained by a passionate developer. Your support is highly valued, and any contributions — whether through issues, bug reports, feature requests, or pull requests — are more than welcome!

Here is the link to the [GitHub repo](https://github.com/stefanodangelo/mdify).

## 4. License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).