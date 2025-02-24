![PyPI](https://img.shields.io/pypi/v/mdify?color=red)
![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY%20NC%204.0-lightgrey)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14795743.svg)](https://doi.org/10.5281/zenodo.14795743)

# MDify: Convert any document to Markdown  

**MDify** is a powerful Python library for converting documents into clean, structured Markdown.

Unlike other tools, **MDify** can accurately extract **tables, charts, and images**, even offering the option to save them separately for further use. \
This is particularly useful when working with documents like financial statements, spreadsheets, and data-rich reports, which usually have lots of tables and images. \
MDify categorizes images into general pictures and charts and extracts tables of any kind, even complex ones with merged cells and sparse data.

Whether you're working with *research papers*, *reports*, or *general documents*, MDify ensures the data is extracted in a structured, clean, and machine-readable format, making it ideal for tasks like fine-tuning, question answering, and document analysis in the context of Large Language Models (**LLMs**). \
By converting complex PDFs into well-structured Markdown, this tool helps streamline the input process for LLM applications, reducing the time spent on manual cleaning and formatting. With features like table extraction, image preservation, and high-quality OCR, MDify is a perfect fit for preparing large volumes of data for AI models.

**IMPORTANT**: Currently this tools only supports PDFs and images (such as text extracts, document scans, etc.) written in English.

## 🚀 Installation  
First, install **MDify** via PyPI:  

```sh
pip install mdify
```


## ⚡ Quickstart  
Convert a document to Markdown with just a few lines of code:
```python
from mdify import DocumentParser

parser = DocumentParser()
parser.parse('PATH_TO_YOUR_DOCUMENT')
```

Or parse multiple documents from one folder at once simply by changing the last line to:
```python
parser.parse_directory('PATH_TO_YOUR_FOLDER')
```

Alternatively, you can also pass the document in bytes to the `parse()` method, but in this case you must also provide the document name and type manually:
```python
with open('PATH_TO_YOUR_DOCUMENT', 'rb') as f:
  document_bytes = f.read()
parser.parse(document_bytes, document_name='YOUR_DOCUMENT_NAME', document_type='pdf')
```

You can then choose the outputs to save using `DocumentParser(save_artifacts=...)`, or you can set the write mode to embedded, placeholder or described by passing the `write_mode` parameter to the `parse()` function.

**NB**: To make the best use of this library and extract meaning from images, use the following code:
```python
from mdify import WriteMode

parser.parse('PATH_TO_YOUR_DOCUMENT', write_mode=WriteMode.DESCRIBED)
```


## 🔹 Key Features  
- ✔️ **Handles complex layouts** - Extracts text, tables, and visual elements with precision
- 🖼️ **Preserves images & charts** - Gives the option to save and reuse extracted visuals for Computer Vision tasks
- 🎯 **Optimized for accuracy** - Combines layout detection and OCR to extract text from documents
- 🤖 **Preprocessing for LLM applications** - Converts documents to Markdown, which is popular for LLM training and fine-tuning tasks
- 🛠️ **Debug mode** - Save intermediate document elements as images for analysis

**NB**:
- The first run will take ~2 minutes to download the necessary models.
- Diagrams are not supported yet, therefore if you use the `DESCRIBED` write mode they may be analyzed incorrectly.


## 📄 Documentation
For more information, please refer to the [official documentation](https://stefanodangelo.github.io/mdify/).


## 🤝 Contributing
MDify is an independent, open-source project developed and maintained by passionate developers. Your support is highly valued, and any contributions — whether through issues, bug reports, feature requests, or pull requests — are more than welcome!

If you are interested in improving this library or adding new features, please don't hesitate to get involved!


## 💖 Support
Being an independent developer, I would much appreciate it if you could\
[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=buy%20me%20a%20coffee&emoji="☕"&slug=stefanodangelo&button_colour=FF5F5F&font_colour=ffffff&font_family=Lato&outline_colour=000000&coffee_colour=FFDD00)](https://www.buymeacoffee.com/stefanodangelo)


Thank you!

## ⚖️ License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

You can find the full text of the license here: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)


## ❞ Citation
If you use this project, please download the citation from [Zenodo](https://doi.org/10.5281/zenodo.14795743) (scroll all the way down and then choose the format you prefer (e.g. Bibtex) from the *Export* dropdown).


## 🔗 Acknowledgments  
This project leverages several open-source repositories for different components:  

- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) – PDF loading
- [Ultralytics](https://github.com/ultralytics/ultralytics) – YOLO-based layout detection
- [Supervision](https://github.com/roboflow/supervision) – Rendering layout elements
- [Surya](https://github.com/VikParuchuri/surya) – Text recognition (primary OCR, though it's currently a bottleneck)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) – Recognizing text headers and titles (solves Surya's issue with them)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) – Table recognition
- [Optimum](https://github.com/huggingface/optimum) – Formula extraction model integration
- [YOLOv10-Document-Layout-Analysis](https://huggingface.co/omoured/YOLOv10-Document-Layout-Analysis) – YOLO model parameters
- [ChartDet](https://huggingface.co/stefanodangelo/chartdet) – Chart detection model parameters
- [DePlot](https://huggingface.co/google/deplot) – Model for chart deconstruction  
- [BLIP Image Captioning](https://huggingface.co/Salesforce/blip-image-captioning-large) – Image captioning model  
- [Pix2Text-MFR](https://huggingface.co/breezedeus/pix2text-mfr) – Formula recognition model  

A huge thanks to the developers and maintainers of these projects!
