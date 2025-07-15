import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # Ensure project root is in path

# -- Project information -----------------------------------------------------
project = "scisbi - Scientific Machine Learning package for Simulation Based Inference documentation"
author = "Csongor Horváth"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # <--- ADD THIS LINE
    "sphinx.ext.napoleon",  # For NumPy/Google style docstrings
    "sphinx_autodoc_typehints",
    "myst_parser",  # For .md support
    "sphinx.ext.viewcode",  # <--- Recommended: Adds links to source code
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True  # <--- ADD THIS LINE

# Optional: Configuration for autosummary to put generated stub files into a specific directory
# This helps keep your main docs directory clean.
autosummary_imported_members = True  # Include members imported from other modules

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
