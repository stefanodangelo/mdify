# Usage

In this page, you will find examples of how to use **MDify**.


## 1. Convert a document
### 1.1 Default parameters
This simple code will convert a document into Markdown:
```python
from mdify import DocumentParser

parser = DocumentParser()
parser.parse('PATH_TO_YOUR_DOCUMENT')
```

By default, no artifacts are saved (i.e. tables, charts, etc.).

### 1.2 Save artifacts
If you want to save the artifacts (i.e. tables, charts, etc.), you can use the `save_artifacts` parameter:
```python
from mdify import DocumentParser, OutputArtifact

parser = DocumentParser(save_artifacts=OutputArtifact.ALL)
parser.parse('PATH_TO_YOUR_DOCUMENT')
```