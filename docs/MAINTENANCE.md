# Documentation Maintenance Guide

This document explains how to maintain and build the documentation for the scisbi project.

## Prerequisites

- Python 3.9+
- Package installed in editable mode: `pip install -e .`
- Documentation dependencies: `pip install -r docs/requirements.txt`

## Building Documentation Locally

### Windows
```bash
cd docs
.\make.bat clean
.\make.bat html
```

### Linux/macOS
```bash
cd docs
make clean
make html
```

The built documentation will be available in `docs/_build/html/index.html`.

## Automatic Documentation Generation

The documentation uses Sphinx with autosummary to automatically generate API documentation from docstrings.

### Key Configuration Files

- `docs/conf.py` - Sphinx configuration
- `docs/api.rst` - API reference structure
- `docs/requirements.txt` - Documentation dependencies
- `.github/workflows/docs.yml` - CI/CD for automatic building

### Generated Files (Do Not Edit)

- `docs/_autosummary/` - Auto-generated API documentation files
- `docs/_build/` - Built HTML documentation

These directories are ignored in `.gitignore` and should not be committed.

## GitHub Actions

The documentation is automatically built on:
- Pull requests to main branch
- Pushes to main branch

The workflow will:
1. Install dependencies
2. Build documentation
3. Upload artifacts (for PRs)
4. Deploy to GitHub Pages (for main branch pushes)

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the package is installed in editable mode (`pip install -e .`)
2. **Missing modules**: Check that all submodules have proper `__init__.py` files
3. **Docstring warnings**: Fix docstring formatting according to NumPy/Google style

### Fixing Docstring Issues

The build may show warnings about docstring formatting. Common fixes:
- Ensure proper indentation in docstrings
- Use blank lines to separate sections
- Follow NumPy or Google docstring conventions

### Path Issues

If Sphinx can't find the package:
1. Check `sys.path` configuration in `docs/conf.py`
2. Verify the package is importable: `python -c "import scisbi"`
3. Ensure you're in the correct directory when building

## Adding New Modules

When adding new modules to the project:
1. The autosummary will automatically detect them
2. No manual changes to `docs/api.rst` are needed
3. Just rebuild the documentation

## Customization

To customize the documentation:
- Edit `docs/index.rst` for the main page
- Add new `.rst` or `.md` files for additional content
- Modify `docs/conf.py` for Sphinx settings
- Update `docs/api.rst` to change API documentation structure
