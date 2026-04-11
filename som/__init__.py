"""SOM + Hebbian cross-modal architecture for CT-Touch.

Provides Self-Organizing Maps with Hebbian cross-modal binding
for learning tactile representations in the MIMo infant simulation.
"""

from som.core import SelfOrganizingMap
from som.preprocessor import TouchPreprocessor
from som.hebbian import HebbianLink, CrossModalNetwork
from som.intrinsic_motivation import IntrinsicMotivation
from som.critical_periods import CriticalPeriodScheduler
from som.som_wrapper import SOMObservationWrapper

__all__ = [
    "SelfOrganizingMap",
    "TouchPreprocessor",
    "HebbianLink",
    "CrossModalNetwork",
    "IntrinsicMotivation",
    "CriticalPeriodScheduler",
    "SOMObservationWrapper",
]
