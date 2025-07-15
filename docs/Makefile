# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS     ?=
SPHINXBUILD    ?= sphinx-build
SOURCEDIR      = .
BUILDDIR       = _build

# Define your package root relative to the docs directory
# Adjust this if your package is nested differently (e.g., in a 'src' folder)
PACKAGE_DIR    = ../your_package_name # <<< CHANGE THIS to your actual package path!

# Define where apidoc will output its RST files
APIDOC_OUTPUT  = $(SOURCEDIR)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html

# Add a clean target for apidoc generated files
clean:
	rm -rf $(BUILDDIR)
	rm -rf $(APIDOC_OUTPUT)/$(PACKAGE_NAME_FOR_APIDOC)*.rst # Clean generated RSTs
	rm -rf $(APIDOC_OUTPUT)/_autosummary # Clean autosummary generated files

# Build HTML documentation
html:
	@echo "Running sphinx-apidoc to generate API stubs..."
	# -o $(APIDOC_OUTPUT): Output to the current docs directory
	# $(PACKAGE_DIR): Path to your Python package
	# --force: Overwrite existing RST files (important for updates)
	# --implicit-namespaces: Use if your package is a PEP 420 namespace package
	# --module-first: Puts module docs before package docs in generated RSTs
	$(SPHINXBUILD) -M apidoc -o "$(APIDOC_OUTPUT)" "$(PACKAGE_DIR)" --force --implicit-namespaces --module-first
	@echo "Building Sphinx HTML documentation..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
