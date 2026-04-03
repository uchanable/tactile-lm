"""CT afferent augmented touch module for MIMo.

Extends TrimeshTouch with CT (C-tactile) afferent mechanoreceptor models,
multi-receptor output (SA-I, FA-I, FA-II, CT), and Gaussian receptive field
spreading.

Usage::

    from ct_touch.ct_augmented_touch import CTAugmentedTouch

    # Override touch_setup in your MIMo environment:
    def touch_setup(self, touch_params):
        self.touch = CTAugmentedTouch(self, touch_params)

References:
    Loken, L. S., et al. (2009). Coding of pleasant touch by unmyelinated
    afferents in humans. Nature Neuroscience, 12(5), 547-548.

    McGlone, F., et al. (2014). Discriminative and affective touch:
    sensing and feeling. Neuron, 82(4), 737-755.
"""

import math
from typing import Dict, Optional

import numpy as np

# Import mimoEnv before mimoTouch to avoid circular import
# (mimoTouch.touch -> mimoEnv.utils -> mimoEnv.__init__ -> ... -> mimoTouch.touch)
import mimoEnv.envs.mimo_env  # noqa: F401 -- side-effect: breaks circular import

from mimoTouch.touch import TrimeshTouch, scale_linear
from mimoEnv.utils import EPS

from ct_touch.skin_map import get_skin_type, SkinType
from ct_touch.developmental import DevelopmentalProfile


class CTAugmentedTouch(TrimeshTouch):
    """TrimeshTouch extended with CT afferent and multi-receptor models.

    This class adds:
    - ``ct_afferent_response``: A touch function that models CT afferent
      firing rate based on stroking velocity (Loken et al. 2009).
    - ``multi_receptor``: A touch function outputting 7-channel per-sensor
      data (SA-I[3] + FA-I[1] + FA-II[1] + CT[1] + normal_force[1]).
    - ``spread_gaussian``: A response function using Gaussian receptive fields.

    Additional Attributes:
        skin_type_map: Maps body_id -> SkinType (hairy/glabrous).
        developmental: Optional DevelopmentalProfile for age-dependent scaling.
        _prev_forces: Stores previous-step forces per body for velocity estimation.
        _contact_velocities: Estimated contact velocities per body for the current step.
    """

    VALID_TOUCH_TYPES = {
        # Inherited from TrimeshTouch
        "force_vector": 3,
        "force_vector_global": 3,
        "normal_force": 3,
        # New touch functions
        "ct_afferent_response": 1,   # scalar firing rate
        "multi_receptor": 7,          # SA-I(3) + FA-I(1) + FA-II(1) + CT(1) + normal(1)
    }

    VALID_RESPONSE_FUNCTIONS = ["nearest", "spread_linear", "spread_gaussian"]

    # CT afferent model parameters (Loken et al. 2009)
    CT_PEAK_VELOCITY = 0.03       # 3 cm/s in m/s
    CT_MAX_FIRING_RATE = 1.0      # normalized max rate
    CT_SIGMA = 0.7                # log-Gaussian width
    CT_MIN_FORCE_THRESHOLD = 0.1  # minimum normal force to trigger CT (N)

    # FA-II tuning parameters
    FA2_PEAK_FREQ = 250.0         # Hz, Pacinian peak sensitivity
    FA2_BANDWIDTH = 100.0         # Hz, half-width of tuning curve

    def __init__(self, env, touch_params, developmental_age: Optional[float] = None):
        """Initialize CTAugmentedTouch.

        Args:
            env: The MIMo environment.
            touch_params: Touch parameter dictionary (same format as TrimeshTouch).
            developmental_age: If provided, enables developmental scaling at this
                age (months, 0-24). None means full adult responses.
        """
        # Build the skin type map before super().__init__ calls add_body
        self._body_name_to_id_cache: Dict[str, int] = {}
        self.skin_type_map: Dict[int, SkinType] = {}

        # Developmental profile
        self.developmental: Optional[DevelopmentalProfile] = None
        if developmental_age is not None:
            self.developmental = DevelopmentalProfile(developmental_age)

        # For velocity estimation
        self._prev_forces: Dict[int, np.ndarray] = {}
        self._contact_velocities: Dict[int, float] = {}

        # Store timestep for velocity calculations
        self._env_ref = env
        self._dt = env.model.opt.timestep * env.frame_skip

        super().__init__(env, touch_params)

        # Build skin type map from body IDs
        self._build_skin_type_map()

    def _build_skin_type_map(self):
        """Populate skin_type_map from body names using the env model."""
        for body_id in self.sensor_scales:
            body_name = self.env.model.body(body_id).name
            self.skin_type_map[body_id] = get_skin_type(body_name)

    def _is_hairy_skin(self, body_id: int) -> bool:
        """Check if the body has hairy skin (and thus CT afferents).

        Args:
            body_id: MuJoCo body ID.

        Returns:
            bool: True if hairy skin.
        """
        return self.skin_type_map.get(body_id, SkinType.HAIRY) == SkinType.HAIRY

    # =========================================================================
    # Velocity estimation
    # =========================================================================

    def _estimate_contact_velocity(self, contact_id: int, body_id: int) -> float:
        """Estimate the tangential velocity of a contact point.

        Uses the contact frame's tangential force change as a proxy for
        velocity. In a full implementation this would track contact point
        positions over time; here we use force-rate-of-change as a
        computationally cheap proxy.

        Args:
            contact_id: MuJoCo contact ID.
            body_id: The sensing body ID.

        Returns:
            float: Estimated velocity in m/s (always >= 0).
        """
        forces = self.get_raw_force(contact_id, body_id)
        # Tangential force magnitude as velocity proxy
        tangential_magnitude = np.sqrt(forces[1] ** 2 + forces[2] ** 2)

        prev = self._prev_forces.get(body_id, None)
        if prev is not None:
            # Rate of change of tangential force ~ sliding velocity
            force_change = abs(tangential_magnitude - prev)
            # Scale factor: convert force-change to approximate velocity
            # This is a heuristic; proper implementation would track contact
            # positions across timesteps.
            velocity = force_change * self._dt * 10.0
        else:
            velocity = 0.0

        self._prev_forces[body_id] = tangential_magnitude
        return max(velocity, 1e-6)  # avoid log(0)

    # =========================================================================
    # CT afferent firing rate model (Loken et al. 2009)
    # =========================================================================

    @staticmethod
    def ct_firing_rate(velocity: float,
                       peak_velocity: float = 0.03,
                       max_rate: float = 1.0,
                       sigma: float = 0.7) -> float:
        """Compute CT afferent firing rate as a function of velocity.

        The response follows a log-Gaussian (inverted-U on log-velocity axis)
        with peak at CT_PEAK_VELOCITY (~3 cm/s).

        Args:
            velocity: Contact velocity in m/s.
            peak_velocity: Velocity at peak firing rate (m/s).
            max_rate: Maximum firing rate (normalized).
            sigma: Width of the log-Gaussian.

        Returns:
            float: Firing rate (0 to max_rate).
        """
        if velocity <= 0:
            return 0.0
        log_ratio = (np.log(velocity) - np.log(peak_velocity)) / sigma
        return max_rate * np.exp(-0.5 * log_ratio ** 2)

    # =========================================================================
    # Touch functions
    # =========================================================================

    def ct_afferent_response(self, contact_id: int, body_id: int) -> np.ndarray:
        """Touch function: CT afferent firing rate.

        Returns a scalar firing rate based on the Loken et al. (2009)
        velocity-dependent inverted-U model. Returns 0 for glabrous skin.

        Args:
            contact_id: MuJoCo contact ID.
            body_id: The sensing body ID.

        Returns:
            np.ndarray: Array of shape (1,) with the CT firing rate.
        """
        # No CT response on glabrous skin
        if not self._is_hairy_skin(body_id):
            return np.zeros(1, dtype=np.float64)

        # Check minimum force threshold
        raw_force = self.get_raw_force(contact_id, body_id)
        normal_force = abs(raw_force[0])
        if normal_force < self.CT_MIN_FORCE_THRESHOLD:
            return np.zeros(1, dtype=np.float64)

        velocity = self._estimate_contact_velocity(contact_id, body_id)
        rate = self.ct_firing_rate(
            velocity,
            peak_velocity=self.CT_PEAK_VELOCITY,
            max_rate=self.CT_MAX_FIRING_RATE,
            sigma=self.CT_SIGMA,
        )

        # Force-dependent gating: CT response scales with gentle force
        # (light touch is optimal, heavy pressure suppresses response)
        force_gate = min(1.0, normal_force / 2.0) * np.exp(-normal_force / 10.0)
        rate *= force_gate

        # Apply developmental scaling if available
        if self.developmental is not None:
            rate *= self.developmental.get_receptor_scale("CT")

        return np.array([rate], dtype=np.float64)

    def multi_receptor(self, contact_id: int, body_id: int) -> np.ndarray:
        """Touch function: Multi-receptor output.

        Outputs a 7-channel vector per contact:
            [0:3] SA-I response (sustained pressure, 3D force vector scaled)
            [3]   FA-I response (velocity-proportional, scalar)
            [4]   FA-II response (vibration/acceleration, scalar)
            [5]   CT response (velocity-tuned inverted-U, scalar)
            [6]   Normal force magnitude (scalar)

        Args:
            contact_id: MuJoCo contact ID.
            body_id: The sensing body ID.

        Returns:
            np.ndarray: Array of shape (7,) with multi-receptor data.
        """
        output = np.zeros(7, dtype=np.float64)
        raw_force = self.get_raw_force(contact_id, body_id)
        normal_force_mag = abs(raw_force[0])

        # --- SA-I: Sustained pressure (slow adaptation) ---
        # Proportional to the force vector, with logarithmic compression
        contact = self.env.data.contact[contact_id]
        force_rot = np.reshape(contact.frame, (3, 3))
        from mimoEnv.utils import rotate_vector_transpose, get_body_rotation
        global_forces = rotate_vector_transpose(raw_force, force_rot)
        body_forces = rotate_vector_transpose(
            global_forces, get_body_rotation(self.env.data, body_id)
        )
        # Log-compress: SA-I adapts slowly but saturates
        sa1_response = np.sign(body_forces) * np.log1p(np.abs(body_forces))
        output[0:3] = sa1_response

        # --- FA-I: Velocity-proportional (Meissner) ---
        # Responds to rate of force change (tangential component proxy)
        velocity = self._estimate_contact_velocity(contact_id, body_id)
        fa1_response = min(velocity * 100.0, 10.0)  # cap at 10
        output[3] = fa1_response

        # --- FA-II: Vibration-tuned (Pacinian, peak ~250 Hz) ---
        # In discrete simulation we approximate as acceleration of normal force
        prev_normal = self._prev_forces.get(body_id, 0.0)
        if isinstance(prev_normal, np.ndarray):
            prev_normal = 0.0
        force_accel = abs(normal_force_mag - prev_normal) / max(self._dt, 1e-6)
        # Tuning: FA-II responds best to rapid changes
        fa2_response = min(force_accel * 0.01, 5.0)
        output[4] = fa2_response

        # --- CT: Velocity-tuned inverted-U ---
        if self._is_hairy_skin(body_id) and normal_force_mag > self.CT_MIN_FORCE_THRESHOLD:
            ct_rate = self.ct_firing_rate(
                velocity,
                peak_velocity=self.CT_PEAK_VELOCITY,
                max_rate=self.CT_MAX_FIRING_RATE,
                sigma=self.CT_SIGMA,
            )
            force_gate = min(1.0, normal_force_mag / 2.0) * np.exp(-normal_force_mag / 10.0)
            ct_rate *= force_gate
            output[5] = ct_rate
        # else: remains 0 for glabrous skin

        # --- Normal force magnitude ---
        output[6] = normal_force_mag

        # Apply developmental scaling
        if self.developmental is not None:
            output[0:3] *= self.developmental.get_receptor_scale("SA1")
            output[3] *= self.developmental.get_receptor_scale("FA1")
            output[4] *= self.developmental.get_receptor_scale("FA2")
            output[5] *= self.developmental.get_receptor_scale("CT")

        return output

    # =========================================================================
    # Response functions
    # =========================================================================

    def spread_gaussian(self, contact_id: int, body_id: int, force: np.ndarray):
        """Response function: Gaussian receptive field spreading.

        Distributes the output force using a Gaussian profile instead of
        linear falloff. CT afferents have larger receptive fields (~35 mm
        diameter) compared to myelinated mechanoreceptors (~2-10 mm).
        The Gaussian sigma is set to 1.5x the sensor scale.

        Args:
            contact_id: MuJoCo contact ID.
            body_id: The sensing body ID.
            force: The raw output force/response array.
        """
        scale = self.sensor_scales[body_id]
        # Search radius: 3x scale for Gaussian (captures ~99.7% of distribution)
        search_radius = 3.0 * scale
        gaussian_sigma = 1.5 * scale

        contact_pos = self.get_contact_position_relative(
            contact_id=contact_id, body_id=body_id
        )
        nearest_sensors, sensor_distances = self.get_sensors_within_distance(
            contact_pos, body_id, search_radius
        )

        if len(nearest_sensors) == 0:
            return

        # Gaussian weights
        weights = np.exp(-0.5 * (sensor_distances / gaussian_sigma) ** 2)
        total_weight = np.sum(weights)

        if total_weight < EPS:
            return

        # Normalize so total force is conserved
        normalized_weights = weights / total_weight

        # Distribute force
        for i, sensor_idx in enumerate(nearest_sensors):
            self.sensor_outputs[body_id][sensor_idx] += force * normalized_weights[i]

    # =========================================================================
    # Override get_touch_obs for velocity tracking reset
    # =========================================================================

    def get_touch_obs(self):
        """Produce touch observations with velocity tracking.

        Overrides parent to reset per-step velocity tracking state.

        Returns:
            np.ndarray: Flattened touch sensor observations.
        """
        # Clear velocity estimates for this step
        self._contact_velocities.clear()

        # Call parent implementation
        result = super().get_touch_obs()

        return result

    # =========================================================================
    # Utility: per-body CT response summary
    # =========================================================================

    def get_ct_summary(self) -> Dict[str, float]:
        """Summarize CT response per body part.

        Returns:
            dict: Maps body_name -> mean absolute sensor output for that body.
                  Only includes bodies that have sensors.
        """
        summary = {}
        for body_id in self.sensor_outputs:
            body_name = self.env.model.body(body_id).name
            output = self.sensor_outputs[body_id]
            summary[body_name] = float(np.mean(np.abs(output)))
        return summary

    def get_skin_type_summary(self) -> Dict[str, str]:
        """Return skin type for each sensed body part.

        Returns:
            dict: Maps body_name -> "hairy" or "glabrous".
        """
        summary = {}
        for body_id in self.sensor_scales:
            body_name = self.env.model.body(body_id).name
            skin_type = self.skin_type_map.get(body_id, SkinType.HAIRY)
            summary[body_name] = skin_type.value
        return summary
