# Minimal Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

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

# Check if required dependencies are available
check-deps:
	@echo "$(BOLD)Checking dependencies...$(RESET)"
	@command -v $(SPHINXBUILD) >/dev/null 2>&1 || { \
		echo "$(RED)Error: $(SPHINXBUILD) not found. Install with: pip install sphinx$(RESET)"; \
		exit 1; \
	}
	@command -v $(SPHINXAPIDOC) >/dev/null 2>&1 || { \
		echo "$(RED)Error: $(SPHINXAPIDOC) not found. Install with: pip install sphinx$(RESET)"; \
		exit 1; \
	}
	@test -d "$(PACKAGE_DIR)" || { \
		echo "$(RED)Error: Package directory '$(PACKAGE_DIR)' not found$(RESET)"; \
		exit 1; \
	}
	@echo "$(GREEN)✓ All dependencies found$(RESET)"

# Clean build artifacts and auto-generated files
clean:
	@echo "$(BOLD)Cleaning documentation build...$(RESET)"
	@echo "$(YELLOW)Removing build directory: $(BUILDDIR)$(RESET)"
	@rm -rf "$(BUILDDIR)"
	@echo "$(YELLOW)Removing auto-generated API documentation...$(RESET)"
	@rm -rf "$(APIDOC_OUTPUT)"
	@rm -f scisbi*.rst modules.rst
	@echo "$(GREEN)✓ Clean completed$(RESET)"

# Deep clean including Python cache and temporary files
clean-all: clean
	@echo "$(BOLD)Performing deep clean...$(RESET)"
	@echo "$(YELLOW)Removing Python cache files...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(YELLOW)Removing temporary files...$(RESET)"
	@find . -type f -name "*~" -delete 2>/dev/null || true
	@find . -type f -name "*.tmp" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Deep clean completed$(RESET)"

# Generate API documentation from source code
apidoc: check-deps
	@echo "$(BOLD)Generating API documentation...$(RESET)"
	@echo "$(YELLOW)Running sphinx-apidoc for package: $(PACKAGE_DIR)$(RESET)"
	@mkdir -p "$(APIDOC_OUTPUT)"
	@$(SPHINXAPIDOC) -f -e -M -o "$(APIDOC_OUTPUT)" "$(PACKAGE_DIR)" \
		--implicit-namespaces --module-first --separate 2>&1 || { \
		echo "$(RED)Error: sphinx-apidoc failed$(RESET)"; \
		exit 1; \
	}
	@echo "$(GREEN)✓ API documentation generated in $(APIDOC_OUTPUT)$(RESET)"

# Build HTML documentation
html: check-deps apidoc
	@echo "$(BOLD)Building HTML documentation...$(RESET)"
	@echo "$(YELLOW)Source: $(SOURCEDIR)$(RESET)"
	@echo "$(YELLOW)Output: $(BUILDDIR)/html$(RESET)"
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) 2>&1 || { \
		echo "$(RED)Error: Sphinx build failed$(RESET)"; \
		echo "$(YELLOW)Check the output above for specific error messages$(RESET)"; \
		exit 1; \
	}
	@echo "$(GREEN)✓ HTML documentation built successfully$(RESET)"
	@echo "$(BLUE)Open $(BUILDDIR)/html/index.html in your browser$(RESET)"

# Build with live reload (development mode)
livehtml: check-deps apidoc
	@echo "$(BOLD)Starting live HTML build...$(RESET)"
	@command -v sphinx-autobuild >/dev/null 2>&1 || { \
		echo "$(YELLOW)Installing sphinx-autobuild...$(RESET)"; \
		pip install sphinx-autobuild; \
	}
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) \
		--host 0.0.0.0 --port 8000 --open-browser

# Debug build with maximum verbosity
debug: check-deps apidoc
	@echo "$(BOLD)Building with debug output...$(RESET)"
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		-v -v -v -T --keep-going $(SPHINXOPTS) 2>&1

# Quick test build to check for syntax errors
test-build: check-deps
	@echo "$(BOLD)Running test build...$(RESET)"
	@$(SPHINXBUILD) -b dummy "$(SOURCEDIR)" "$(BUILDDIR)/test" \
		-W -q $(SPHINXOPTS) 2>&1 && \
	echo "$(GREEN)✓ Test build passed$(RESET)" || { \
		echo "$(RED)✗ Test build failed$(RESET)"; \
		exit 1; \
	}

# Catch-all target for Sphinx make mode commands
%: check-deps
	@echo "$(BOLD)Running Sphinx in make mode: $@$(RESET)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
