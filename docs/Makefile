# Minimal Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html check-deps


define check_sphinxbuild
	@command -v $(SPHINXBUILD) >/dev/null 2>&1 || { \
		echo; \
		echo "The 'sphinx-build' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the 'sphinx-build' executable."; \
		echo; \
		exit 1; \
	}
endef

help:
	@echo "Sphinx documentation build targets:"
	@echo "  html       Build HTML documentation"
	@echo "  clean      Remove build files"
	@echo "  check-deps Check dependencies"

clean:
	$(call check_sphinxbuild)
	rm -rf $(BUILDDIR)

html:
	$(call check_sphinxbuild)
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

check-deps:
	$(call check_sphinxbuild)
	@python -c "import sphinx, furo, myst_parser; print('All dependencies available')"
