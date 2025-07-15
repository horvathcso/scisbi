# Documentation Build System

This directory contains the Sphinx documentation build system for the scisbi project, with enhanced functionality for both local development and CI/CD pipelines.

## Quick Start

### Windows Users
```batch
# Show available commands
.\make.bat help

# Clean and build documentation
.\make.bat clean
.\make.bat html
```

### Linux/macOS Users (or Windows with Make)
```bash
# Show available commands
make help

# Clean and build documentation
make clean
make html
```

### PowerShell Users (Alternative)
```powershell
# Show available commands
.\build.ps1 help

# Clean and build documentation
.\build.ps1 clean
.\build.ps1 html
```

## Available Commands

| Command      | Description                                           |
|--------------|-------------------------------------------------------|
| `help`       | Show available commands and environment information   |
| `clean`      | Remove build files and auto-generated API docs       |
| `clean-all`  | Deep clean including Python cache and temp files     |
| `apidoc`     | Generate API documentation from source code          |
| `html`       | Build complete HTML documentation                     |
| `debug`      | Build with maximum verbosity for troubleshooting     |
| `check-deps` | Verify required dependencies are installed           |
| `test-build` | Quick syntax check without full build                |

## Build System Components

### 1. Makefile (Linux/macOS/GitHub Actions)
- Enhanced Makefile with comprehensive error handling
- Colored output for better readability
- Automatic API documentation generation
- GitHub Actions compatible

### 2. make.bat (Windows Batch)
- Windows-native batch file with same functionality as Makefile
- Enhanced error messages and colored output
- Compatible with Windows Command Prompt and PowerShell

### 3. build.ps1 (PowerShell Alternative)
- PowerShell script for Windows users who prefer PowerShell
- Cross-platform PowerShell Core compatible
- Rich error handling and colored output

## Dependencies

The build system requires:
- **Python 3.8+**
- **Sphinx** (`pip install sphinx`)
- **sphinx-apidoc** (included with Sphinx)
- **Project dependencies** (from `requirements.txt`)

## Environment Configuration

The build system automatically detects and configures:
- `SPHINXBUILD` - Path to sphinx-build executable
- `SPHINXAPIDOC` - Path to sphinx-apidoc executable
- `PACKAGE_DIR` - Location of Python package (`../src/scisbi`)
- `BUILDDIR` - Output directory for built documentation (`_build`)
- `APIDOC_OUTPUT` - Directory for auto-generated API docs (`_autosummary`)

## GitHub Actions Integration

The documentation automatically builds on:
- Pull requests to `main` branch
- Pushes to `main` branch

The build process:
1. Installs dependencies
2. Runs dependency checks
3. Cleans previous builds
4. Generates API documentation
5. Builds HTML documentation
6. Uploads artifacts
7. Deploys to GitHub Pages (on main branch)

## Troubleshooting

### Build Failures
1. Run `make check-deps` (or `.\make.bat check-deps`) to verify dependencies
2. Use `make debug` (or `.\make.bat debug`) for verbose output
3. Check that the package directory exists: `../src/scisbi`

### Common Issues
- **"sphinx-build not found"**: Install Sphinx with `pip install sphinx`
- **"Package directory not found"**: Ensure you're in the `docs/` directory
- **"Permission denied"**: On Windows, run as Administrator if needed

### GitHub Actions Debugging
If GitHub Actions builds fail:
1. Check the "Debug build (on failure)" step for detailed error output
2. Verify all dependencies are listed in `requirements.txt` and `docs/requirements.txt`
3. Ensure the package structure matches the expected layout

## Output

Built documentation is available in:
- **HTML**: `_build/html/index.html`
- **API Documentation**: `_autosummary/` (auto-generated)

## Development Workflow

For documentation development:
1. Edit documentation files in the `docs/` directory
2. Run `make html` (or `.\make.bat html`) to build
3. Open `_build/html/index.html` in your browser
4. Repeat as needed

For continuous development, you can install `sphinx-autobuild` and use:
```bash
pip install sphinx-autobuild
make livehtml  # Available in Makefile only
```

This will start a local server with live reload functionality.
