# src/scisbi/__init__.py
__version__ = "0.0.1"


import os
import importlib

__all__ = []

# Import modules directly within the current directory
for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)
        __all__.extend([name for name in dir(module) if not name.startswith("_")])
        del module

# Import from subdirectories
for dirname in os.listdir(os.path.dirname(__file__)):
    subdirectory = os.path.join(os.path.dirname(__file__), dirname)
    if os.path.isdir(subdirectory) and os.path.exists(
        os.path.join(subdirectory, "__init__.py")
    ):
        try:
            sub_module = importlib.import_module(f".{dirname}", package=__name__)
            __all__.extend(
                [name for name in dir(sub_module) if not name.startswith("_")]
            )
            del sub_module
        except ImportError as e:
            print(f"Warning: Could not import submodule '{dirname}'. Reason: {e}")

# Remove duplicates from __all__ (in case of naming conflicts)
__all__ = sorted(list(set(__all__)))
