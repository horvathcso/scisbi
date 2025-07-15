# scisbi/inference/__init__.py

# Import the inference classes to make them available under the scisbi.inference namespace
from .ABC import ABCRejectionSampling, ABCPosterior
from .ABCMCMC import ABCMCMC, ABCMCMCPosterior
from .ABCSMC import ABCSMC, ABCSMCPosterior

# Try to import neural network based methods (require torch)
try:
    from .JANA import JANA, JANAPosterior
    from .Simformer import Simformer, SimformerPosterior, SimformerTransformer

    NEURAL_METHODS_AVAILABLE = True
except ImportError:
    # If torch is not available, these methods won't be available
    NEURAL_METHODS_AVAILABLE = False


__all__ = [
    "ABCRejectionSampling",
    "ABCPosterior",
    "ABCMCMC",
    "ABCMCMCPosterior",
    "ABCSMC",
    "ABCSMCPosterior",
]

# Add neural methods to __all__ if available
if NEURAL_METHODS_AVAILABLE:
    __all__.extend(
        [
            "JANA",
            "JANAPosterior",
            "Simformer",
            "SimformerPosterior",
            "SimformerTransformer",
        ]
    )
