from pathlib import Path
import setuptools

VERSION = "0.1.0"  # PEP-440
NAME = "mdify"
INSTALL_REQUIRES = open('./requirements.txt', 'r').read().split('\n')
AUTHOR = r"Stefano D'Angelo"
URL = "https://github.com/stefanodangelo/mdify"

setuptools.setup(
    name=NAME,
    version=VERSION,
    description="A powerful tool to extract text, tables, charts, and formulas from documents and convert them into Markdown format, ideal to improve LLM's accuracy and for versatile document processing.",
    url=URL,
    project_urls={
        "Source Code": URL,
    },
    author=AUTHOR,
    license="CC BY-NC 4.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",
        "Framework :: MkDocs",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires=">=3.8",
    # Requirements
    install_requires=INSTALL_REQUIRES,
    packages=["mdify"],
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
)