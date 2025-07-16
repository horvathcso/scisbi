# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add the src directory to the path so Sphinx can find the scisbi package
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "scisbi"
copyright = "2025, Csongor Horváth"
author = "Csongor Horváth"
release = "0.0.1"
version = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Extension configuration -------------------------------------------------
# autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} v{version}"
