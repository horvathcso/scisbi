import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # Ensure project root is in path

# -- Project information -----------------------------------------------------
project = "YourProjectName"
author = "Your Name"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For NumPy/Google style docstrings
    "sphinx_autodoc_typehints",
    "myst_parser",  # For .md support
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
