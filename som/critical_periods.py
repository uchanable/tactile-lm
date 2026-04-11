"""Critical period scheduling for SOM development.

Models the modality-specific critical periods observed in infant
sensory development. During a critical period, the SOM learning
rate and neighborhood radius are boosted, enabling rapid
reorganization. After the critical period closes, learning slows
and the representation stabilizes.

This interacts with ct_touch/developmental.py for receptor-level
maturation, providing a complementary cortical-level scheduling.

Reference:
    Hensch, T. K. (2005). Critical period plasticity in local
    cortical circuits. Nature Reviews Neuroscience.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from som.core import SelfOrganizingMap


# Default critical period windows (in simulated months)
# Based on developmental neuroscience literature:
#   - Tactile: earliest to mature, critical period closes first
#   - Proprioceptive: slightly later
#   - Visual: latest (foveal acuity reaches adult levels ~12mo)
#   - Cross-modal (Hebbian): peaks during multimodal integration period
DEFAULT_CRITICAL_PERIODS = {
    "tactile_disc": {"onset": 0, "peak": 3, "offset": 12},
    "tactile_aff": {"onset": 0, "peak": 2, "offset": 10},
    "proprio":      {"onset": 2, "peak": 6, "offset": 14},
    "visual":       {"onset": 4, "peak": 8, "offset": 18},
    "hebbian":      {"onset": 3, "peak": 9, "offset": 18},
}

# Learning rate multipliers during critical period phases
LR_MULTIPLIERS = {
    "before_onset": 0.3,   # Some learning before critical period
    "rising": 1.0,         # Ramps up onset -> peak
    "peak": 1.5,           # Boosted at peak
    "falling": 1.0,        # Ramps down peak -> offset
    "after_offset": 0.1,   # Minimal learning after closure
}


class CriticalPeriodScheduler:
    """Schedules SOM learning parameters based on developmental age.

    Maps simulated developmental age (months) to learning rate and
    neighborhood radius multipliers for each SOM modality.

    Attributes:
        periods: Dict mapping modality name -> critical period spec.
        steps_per_month: Training steps per simulated month.
    """

    def __init__(
        self,
        periods: Optional[Dict[str, dict]] = None,
        steps_per_month: int = 50_000,
    ):
        """Initialize scheduler.

        Args:
            periods: Dict mapping modality name -> {"onset", "peak", "offset"}
                in months. Uses DEFAULT_CRITICAL_PERIODS if None.
            steps_per_month: Number of training steps per simulated month.
                Controls the mapping from training time to developmental age.
        """
        self.periods = periods or DEFAULT_CRITICAL_PERIODS
        self.steps_per_month = steps_per_month

    def age_from_step(self, step: int) -> float:
        """Convert training step to simulated age in months.

        Args:
            step: Current training step.

        Returns:
            float: Simulated age in months.
        """
        return step / self.steps_per_month

    def get_multiplier(self, modality: str, step: int) -> float:
        """Get learning rate multiplier for a modality at given step.

        Returns a value in [0.1, 1.5] based on whether the modality
        is in its critical period.

        Args:
            modality: SOM modality name (e.g., "tactile_disc").
            step: Current training step.

        Returns:
            float: Learning rate multiplier.
        """
        if modality not in self.periods:
            return 1.0

        age = self.age_from_step(step)
        period = self.periods[modality]
        onset = period["onset"]
        peak = period["peak"]
        offset = period["offset"]

        if age < onset:
            return LR_MULTIPLIERS["before_onset"]
        elif age < peak:
            # Linear ramp from before_onset to peak
            t = (age - onset) / max(peak - onset, 1e-6)
            return LR_MULTIPLIERS["before_onset"] + t * (
                LR_MULTIPLIERS["peak"] - LR_MULTIPLIERS["before_onset"]
            )
        elif age < offset:
            # Linear ramp from peak to after_offset
            t = (age - peak) / max(offset - peak, 1e-6)
            return LR_MULTIPLIERS["peak"] + t * (
                LR_MULTIPLIERS["after_offset"] - LR_MULTIPLIERS["peak"]
            )
        else:
            return LR_MULTIPLIERS["after_offset"]

    def get_sigma_multiplier(self, modality: str, step: int) -> float:
        """Get neighborhood radius multiplier.

        Follows the same profile as learning rate: wider neighborhood
        during critical period, narrower after closure.

        Args:
            modality: SOM modality name.
            step: Current training step.

        Returns:
            float: Sigma multiplier.
        """
        # Same profile as lr multiplier
        return self.get_multiplier(modality, step)

    def apply_to_som(self, som: SelfOrganizingMap, modality: str, step: int):
        """Apply critical period scaling to a SOM's current parameters.

        Modulates the SOM's effective learning rate and sigma by the
        critical period multiplier. Call this before each SOM update.

        Note: This temporarily overrides the SOM's internal decay schedule.
        The SOM's _step counter is NOT modified.

        Args:
            som: The SOM to modulate.
            modality: Modality name for period lookup.
            step: Current global training step.
        """
        mult = self.get_multiplier(modality, step)

        # Store original values on first call
        if not hasattr(som, '_base_initial_lr'):
            som._base_initial_lr = som.initial_lr
            som._base_initial_sigma = som.initial_sigma

        # Modulate
        som.initial_lr = som._base_initial_lr * mult
        som.initial_sigma = som._base_initial_sigma * mult

    def get_developmental_profile(self, step: int) -> Dict[str, float]:
        """Get all modality multipliers at current step.

        Useful for logging and visualization.

        Args:
            step: Current training step.

        Returns:
            Dict mapping modality -> multiplier.
        """
        age = self.age_from_step(step)
        profile = {"age_months": age}
        for modality in self.periods:
            profile[f"{modality}_lr_mult"] = self.get_multiplier(modality, step)
        return profile

    def is_in_critical_period(self, modality: str, step: int) -> bool:
        """Check if a modality is currently in its critical period.

        Args:
            modality: Modality name.
            step: Current training step.

        Returns:
            bool: True if between onset and offset.
        """
        if modality not in self.periods:
            return False
        age = self.age_from_step(step)
        period = self.periods[modality]
        return period["onset"] <= age <= period["offset"]

    def get_state(self) -> dict:
        """Serialize scheduler state."""
        return {
            "periods": self.periods,
            "steps_per_month": self.steps_per_month,
        }
