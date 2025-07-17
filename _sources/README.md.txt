# Documentation Build Guide

This documentation system uses Sphinx with a minimal, clean configuration.

## Quick Start

### Local Development

**Windows:**
```bash
cd docs
make.bat html
```

**Linux/macOS:**
```bash
cd docs
make html
```

### Dependencies

Install required packages:
```bash
pip install -r docs/requirements.txt
```

Dependencies include:
- `sphinx>=5.0` - Documentation builder
- `furo` - Clean, modern theme
- `myst-parser` - Markdown support

## Structure

```
docs/
├── conf.py           # Sphinx configuration
├── index.rst         # Main documentation index
├── overview.rst      # Project overview
├── quickstart.rst    # Getting started guide
├── api.rst          # API reference
├── requirements.txt  # Documentation dependencies
├── Makefile         # Linux/macOS build commands
├── make.bat         # Windows build commands
├── _static/         # Static files (CSS, images)
├── _templates/      # Custom Jinja2 templates
└── _build/          # Generated documentation (auto-created)
```

## Available Commands

- `make html` / `make.bat html` - Build HTML documentation
- `make clean` / `make.bat clean` - Remove build files
- `make check-deps` / `make.bat check-deps` - Verify dependencies

## GitHub Actions

Documentation is automatically built and deployed via GitHub Actions:
- **Trigger:** Push or PR to `main` branch
- **Build:** Runs on Ubuntu with Python 3.11
- **Deploy:** GitHub Pages (main branch only)

## Features

- **Auto-generated API docs** from docstrings
- **Modern Furo theme** with dark/light mode
- **Markdown support** via MyST parser
- **Search functionality** built-in
- **Mobile responsive** design
- **Fast builds** with minimal warnings

## Customization

- **Theme settings:** Edit `html_*` options in `conf.py`
- **Static files:** Add CSS/JS to `_static/`
- **Templates:** Customize layouts in `_templates/`
- **Extensions:** Add Sphinx extensions to `extensions` list

The system is designed to be simple, maintainable, and produce clean, professional documentation with minimal configuration.
