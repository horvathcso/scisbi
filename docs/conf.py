import os
import sys

# Add the src directory to the path so Sphinx can find the scisbi package
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "scisbi - Scientific Machine Learning package for Simulation Based Inference documentation"
author = "Csongor Horváth"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # For NumPy/Google style docstrings
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx.ext.intersphinx",  # Link to other project docs
    "sphinx_autodoc_typehints",
    "myst_parser",  # For .md support
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True
autosummary_imported_members = True

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
