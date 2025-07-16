# Minimal Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html check-deps

help:
	@echo "Sphinx documentation build targets:"
	@echo "  html       Build HTML documentation"
	@echo "  clean      Remove build files"
	@echo "  check-deps Check dependencies"

clean:
	rm -rf $(BUILDDIR)

html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

check-deps:
	@python -c "import sphinx, furo, myst_parser; print('All dependencies available')"
