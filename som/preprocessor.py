"""Touch observation preprocessor for SOM input.

Transforms raw multi-receptor touch observations (n_sensors x 7 channels)
into body-part-pooled feature vectors suitable for SOM learning.

Channel layout (from ct_augmented_touch.py multi_receptor):
    [0:3] SA-I response (3D body-frame force, log-compressed)
    [3]   FA-I response (velocity-proportional scalar)
    [4]   FA-II response (vibration/acceleration scalar)
    [5]   CT response (velocity-tuned inverted-U scalar)
    [6]   Normal force magnitude (scalar)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ct_touch.skin_map import get_skin_type, SkinType


# Channel indices in the 7-channel multi_receptor output
CH_SA1 = slice(0, 3)  # 3D
CH_FA1 = 3            # scalar
CH_FA2 = 4            # scalar
CH_CT = 5             # scalar
CH_NORMAL = 6         # scalar

# Discriminative channels: SA-I(3) + FA-I(1) + FA-II(1) + Normal(1) = 6
DISC_DIM_PER_BODY = 6
# Affective channel: CT(1)
AFF_DIM_PER_BODY = 1


class TouchPreprocessor:
    """Preprocesses multi-receptor touch into body-part-pooled SOM inputs.

    Transforms per-sensor 7-channel output into two feature vectors:
    - discriminative: (n_bodies, 6) — SA-I, FA-I, FA-II, normal force
    - affective: (n_bodies, 1) — CT response only

    The body-part pooling preserves somatotopic structure while reducing
    dimensionality from ~21,000 to ~100 dimensions.

    Attributes:
        body_ids: Sorted list of body IDs with touch sensors.
        body_names: Corresponding body names.
        skin_types: SkinType for each body.
        n_bodies: Number of sensing bodies.
        disc_dim: Total discriminative input dimension.
        aff_dim: Total affective input dimension.
    """

    def __init__(self, touch_module, env):
        """Initialize from a touch module instance.

        Args:
            touch_module: CTAugmentedTouch (or TrimeshTouch) instance.
                Must have .meshes, .sensor_outputs attributes.
            env: MIMoEnv instance (for body name lookup).
        """
        self.touch = touch_module
        self.env = env

        # Discover bodies in deterministic order (same as flatten_sensor_dict)
        self.body_ids: List[int] = sorted(touch_module.meshes.keys())
        self.body_names: List[str] = [
            env.model.body(bid).name for bid in self.body_ids
        ]
        self.skin_types: List[SkinType] = [
            get_skin_type(name) for name in self.body_names
        ]
        self.n_bodies = len(self.body_ids)

        # Precompute hairy body mask for CT channel
        self._hairy_mask = np.array(
            [st == SkinType.HAIRY for st in self.skin_types], dtype=bool
        )

        # Output dimensions
        self.disc_dim = self.n_bodies * DISC_DIM_PER_BODY
        self.aff_dim = self.n_bodies * AFF_DIM_PER_BODY

        # Running statistics for online normalization
        self._disc_mean = np.zeros(self.disc_dim)
        self._disc_var = np.ones(self.disc_dim)
        self._aff_mean = np.zeros(self.aff_dim)
        self._aff_var = np.ones(self.aff_dim)
        self._n_samples = 0

    @property
    def is_multi_receptor(self) -> bool:
        """Whether the touch module uses 7-channel multi_receptor output."""
        return self.touch.touch_size == 7

    def process(
        self, sensor_outputs: Optional[Dict[int, np.ndarray]] = None,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Process touch sensor outputs into SOM-ready feature vectors.

        Should be called after env.touch.get_touch_obs() has populated
        sensor_outputs.

        Args:
            sensor_outputs: Dict[body_id -> (n_sensors, touch_size)].
                If None, reads from self.touch.sensor_outputs.
            normalize: Whether to apply running normalization.

        Returns:
            Dict with keys:
                "discriminative": np.ndarray of shape (disc_dim,)
                "affective": np.ndarray of shape (aff_dim,)
                "disc_2d": np.ndarray of shape (n_bodies, 6) — before flatten
                "aff_2d": np.ndarray of shape (n_bodies, 1) — before flatten
        """
        if sensor_outputs is None:
            sensor_outputs = self.touch.sensor_outputs

        disc_features = np.zeros((self.n_bodies, DISC_DIM_PER_BODY))
        aff_features = np.zeros((self.n_bodies, AFF_DIM_PER_BODY))

        for i, body_id in enumerate(self.body_ids):
            body_data = sensor_outputs.get(body_id)
            if body_data is None or body_data.shape[0] == 0:
                continue

            if self.is_multi_receptor:
                self._extract_multi_receptor(body_data, i, disc_features, aff_features)
            else:
                self._extract_force_vector(body_data, i, disc_features)

        # Flatten
        disc_flat = disc_features.ravel()
        aff_flat = aff_features.ravel()

        # Update running stats and normalize
        if normalize:
            self._update_stats(disc_flat, aff_flat)
            disc_flat = self._normalize(disc_flat, self._disc_mean, self._disc_var)
            aff_flat = self._normalize(aff_flat, self._aff_mean, self._aff_var)

        return {
            "discriminative": disc_flat,
            "affective": aff_flat,
            "disc_2d": disc_features,
            "aff_2d": aff_features,
        }

    def _extract_multi_receptor(
        self,
        body_data: np.ndarray,
        body_idx: int,
        disc_out: np.ndarray,
        aff_out: np.ndarray,
    ):
        """Extract features from 7-channel multi_receptor data.

        Uses mean pooling across sensors for each body part.
        """
        # body_data shape: (n_sensors, 7)
        sa1 = body_data[:, CH_SA1]       # (n_sensors, 3)
        fa1 = body_data[:, CH_FA1]       # (n_sensors,)
        fa2 = body_data[:, CH_FA2]       # (n_sensors,)
        ct = body_data[:, CH_CT]         # (n_sensors,)
        normal = body_data[:, CH_NORMAL] # (n_sensors,)

        # Discriminative: mean pool [SA-I(3), FA-I(1), FA-II(1), Normal(1)]
        disc_out[body_idx, 0:3] = sa1.mean(axis=0)
        disc_out[body_idx, 3] = fa1.mean()
        disc_out[body_idx, 4] = fa2.mean()
        disc_out[body_idx, 5] = normal.mean()

        # Affective: mean CT activation
        aff_out[body_idx, 0] = ct.mean()

    def _extract_force_vector(
        self,
        body_data: np.ndarray,
        body_idx: int,
        disc_out: np.ndarray,
    ):
        """Extract features from 3-channel force_vector data.

        For CT OFF / baseline conditions. Maps 3D force to discriminative
        channels, leaving FA-I/FA-II/CT as zero.
        """
        # body_data shape: (n_sensors, 3)
        disc_out[body_idx, 0:3] = body_data.mean(axis=0)
        # FA-I, FA-II, Normal remain zero (not available in force_vector mode)

    def _update_stats(self, disc: np.ndarray, aff: np.ndarray):
        """Welford's online algorithm for running mean and variance."""
        self._n_samples += 1
        n = self._n_samples

        # Discriminative
        delta = disc - self._disc_mean
        self._disc_mean += delta / n
        delta2 = disc - self._disc_mean
        self._disc_var += (delta * delta2 - self._disc_var) / n

        # Affective
        delta = aff - self._aff_mean
        self._aff_mean += delta / n
        delta2 = aff - self._aff_mean
        self._aff_var += (delta * delta2 - self._aff_var) / n

    @staticmethod
    def _normalize(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Standardize using running statistics."""
        std = np.sqrt(var + 1e-8)
        return (x - mean) / std

    def get_body_contact_summary(
        self, sensor_outputs: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Get per-body contact activation summary.

        Useful for behavioral pattern analysis (which body regions
        are contacted most frequently).

        Returns:
            Dict mapping body_name -> total activation magnitude.
        """
        if sensor_outputs is None:
            sensor_outputs = self.touch.sensor_outputs

        summary = {}
        for i, body_id in enumerate(self.body_ids):
            body_data = sensor_outputs.get(body_id)
            if body_data is None:
                summary[self.body_names[i]] = 0.0
            else:
                summary[self.body_names[i]] = float(np.sum(np.abs(body_data)))
        return summary

    def get_ct_by_region(
        self, sensor_outputs: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Get CT activation per body region.

        Only meaningful for multi_receptor mode. Returns zero for
        glabrous skin regions.

        Returns:
            Dict mapping body_name -> mean CT activation.
        """
        if not self.is_multi_receptor:
            return {name: 0.0 for name in self.body_names}

        if sensor_outputs is None:
            sensor_outputs = self.touch.sensor_outputs

        ct_map = {}
        for i, body_id in enumerate(self.body_ids):
            body_data = sensor_outputs.get(body_id)
            if body_data is None or body_data.shape[0] == 0:
                ct_map[self.body_names[i]] = 0.0
            else:
                ct_map[self.body_names[i]] = float(body_data[:, CH_CT].mean())
        return ct_map
