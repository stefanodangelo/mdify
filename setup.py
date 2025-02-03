from pathlib import Path
from setuptools import setup, find_packages

VERSION = "0.1.5"  # PEP-440
NAME = "mdify"
INSTALL_REQUIRES = open('./requirements.txt', 'r').read().split('\n')
AUTHOR = r"Stefano D'Angelo"
URL = "https://github.com/stefanodangelo/mdify"

setup(
    name=NAME,
    version=VERSION,
    description="A powerful tool to extract text, tables, charts, and formulas from documents and convert them into Markdown format, ideal to improve LLM's accuracy and for versatile document processing.",
    url=URL,
    project_urls={
        "GitHub Repository": URL,
        "Documentation": "https://stefanodangelo.github.io/mdify/",
    },
    author=AUTHOR,
    license="CC BY-NC 4.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
    packages=find_packages(include=["mdify", "mdify.*"]),
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
)