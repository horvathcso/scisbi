# Enhanced Makefile for Sphinx documentation
# Compatible with GitHub Actions and local development
# Provides comprehensive debugging output and error handling

# Build configuration variables
SPHINXOPTS     ?= -v -W --keep-going
SPHINXBUILD    ?= sphinx-build
SPHINXAPIDOC   ?= sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build

# Package and API documentation paths
PACKAGE_DIR   = ../src/scisbi
APIDOC_OUTPUT = _autosummary

# Colors for output (works in most terminals including GitHub Actions)
RESET   = \033[0m
BOLD    = \033[1m
RED     = \033[31m
GREEN   = \033[32m
YELLOW  = \033[33m
BLUE    = \033[34m

# Default target
.DEFAULT_GOAL := help

# Declare phony targets
.PHONY: help clean clean-all apidoc html livehtml debug check-deps install-deps test-build

# Help target - shows available commands
help:
	@echo "$(BOLD)Sphinx Documentation Build System$(RESET)"
	@echo ""
	@echo "$(BOLD)Available targets:$(RESET)"
	@echo "  $(GREEN)help$(RESET)        - Show this help message"
	@echo "  $(GREEN)clean$(RESET)       - Remove build files and auto-generated API docs"
	@echo "  $(GREEN)clean-all$(RESET)   - Deep clean including cache and temporary files"
	@echo "  $(GREEN)apidoc$(RESET)      - Generate API documentation from source code"
	@echo "  $(GREEN)html$(RESET)        - Build HTML documentation"
	@echo "  $(GREEN)livehtml$(RESET)    - Build HTML with live reload (requires sphinx-autobuild)"
	@echo "  $(GREEN)debug$(RESET)       - Build with maximum verbosity for debugging"
	@echo "  $(GREEN)check-deps$(RESET)  - Check if required dependencies are installed"
	@echo "  $(GREEN)test-build$(RESET)  - Quick test build to check for errors"
	@echo ""
	@echo "$(BOLD)Environment:$(RESET)"
	@echo "  SPHINXBUILD   = $(SPHINXBUILD)"
	@echo "  SPHINXAPIDOC  = $(SPHINXAPIDOC)"
	@echo "  PACKAGE_DIR   = $(PACKAGE_DIR)"
	@echo "  BUILDDIR      = $(BUILDDIR)"

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
