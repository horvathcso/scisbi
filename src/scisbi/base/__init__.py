# scisbi/base/__init__.py

# Import the base classes to make them available under the scisbi.base namespace
from .inference import BaseInferenceAlgorithm
from .simulator import BaseSimulator
from .summary_statistic import BaseSummaryStatistic
from .workflow import BaseWorkflow

# Define __all__ for the base submodule
__all__ = [
    "BaseInferenceAlgorithm",
    "BaseSimulator",
    "BaseSummaryStatistic",
    "BaseWorkflow",
]
