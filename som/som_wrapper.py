"""Gymnasium wrapper for SOM-based observation transformation.

Wraps any MIMo environment with touch sensing, replacing raw touch
observations with SOM activation maps. Supports both CT ON
(multi_receptor) and CT OFF (force_vector) modes.

Usage:
    base_env = CTReachEnv()
    wrapped = SOMObservationWrapper(base_env, som_config)
    model = PPO("MultiInputPolicy", wrapped)
"""

import gymnasium
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from som.core import SelfOrganizingMap
from som.preprocessor import TouchPreprocessor
from som.hebbian import CrossModalNetwork


# Default SOM configuration for CT-Touch
DEFAULT_SOM_CONFIG = {
    # SOM grid sizes
    "disc_grid": (15, 15),   # Discriminative SOM: 225 neurons
    "aff_grid": (10, 10),    # Affective SOM: 100 neurons
    "proprio_grid": (10, 10),  # Proprioceptive SOM: 100 neurons

    # Learning parameters
    "initial_lr": 0.5,
    "final_lr": 0.01,
    "initial_sigma": None,   # Defaults to max(grid)/2
    "final_sigma": 0.5,
    "decay_steps": 200_000,

    # Hebbian parameters
    "hebbian_eta": 0.01,
    "hebbian_decay": 0.001,

    # What to include in the observation
    "include_proprio_som": True,
    "include_raw_proprio": True,  # Keep original proprio alongside SOM

    # SOM learning during evaluation
    "learn_during_eval": False,
}


class SOMObservationWrapper(gymnasium.Wrapper):
    """Wraps MIMo environment with SOM-based touch representation.

    Replaces the flat touch observation with SOM activation maps.
    Optionally adds a proprioceptive SOM. Performs SOM and Hebbian
    learning at each step.

    The observation space changes from:
        {"observation": proprio, "touch": flat_touch}
    to:
        {"observation": proprio, "som_repr": concatenated_som_activations}

    Attributes:
        preprocessor: TouchPreprocessor for channel splitting.
        network: CrossModalNetwork managing SOMs and Hebbian links.
        som_config: Configuration dictionary.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        som_config: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        """Initialize SOM wrapper.

        Args:
            env: Base MIMo environment with touch enabled.
            som_config: SOM configuration (merged with DEFAULT_SOM_CONFIG).
            seed: Random seed for SOM initialization.
        """
        super().__init__(env)

        self.som_config = {**DEFAULT_SOM_CONFIG, **(som_config or {})}
        self._rng = np.random.default_rng(seed)
        self._learning_enabled = True

        self.preprocessor: Optional[TouchPreprocessor] = None
        self.network: Optional[CrossModalNetwork] = None
        self._initialized = False

        # Eagerly initialize if touch is already set up (MIMo inits touch
        # in __init__, so it's available before the first reset).
        if hasattr(env, 'touch') and env.touch is not None:
            self._lazy_init()

    def _lazy_init(self):
        """Initialize SOM components after environment is set up."""
        if self._initialized:
            return

        cfg = self.som_config

        # Initialize preprocessor
        self.preprocessor = TouchPreprocessor(self.env.touch, self.env)

        # Build SOM configs
        som_configs = {
            "tactile_disc": {
                "grid_size": cfg["disc_grid"],
                "input_dim": self.preprocessor.disc_dim,
                "initial_lr": cfg["initial_lr"],
                "final_lr": cfg["final_lr"],
                "initial_sigma": cfg["initial_sigma"],
                "final_sigma": cfg["final_sigma"],
                "decay_steps": cfg["decay_steps"],
                "rng": np.random.default_rng(self._rng.integers(2**31)),
            },
        }

        # Affective SOM only if multi_receptor mode (CT available)
        if self.preprocessor.is_multi_receptor:
            som_configs["tactile_aff"] = {
                "grid_size": cfg["aff_grid"],
                "input_dim": self.preprocessor.aff_dim,
                "initial_lr": cfg["initial_lr"],
                "final_lr": cfg["final_lr"],
                "initial_sigma": cfg["initial_sigma"],
                "final_sigma": cfg["final_sigma"],
                "decay_steps": cfg["decay_steps"],
                "rng": np.random.default_rng(self._rng.integers(2**31)),
            }

        # Proprioceptive SOM
        if cfg["include_proprio_som"]:
            # Get proprio dim from observation space
            proprio_dim = self.env.observation_space["observation"].shape[0]
            som_configs["proprio"] = {
                "grid_size": cfg["proprio_grid"],
                "input_dim": proprio_dim,
                "initial_lr": cfg["initial_lr"],
                "final_lr": cfg["final_lr"],
                "initial_sigma": cfg["initial_sigma"],
                "final_sigma": cfg["final_sigma"],
                "decay_steps": cfg["decay_steps"],
                "rng": np.random.default_rng(self._rng.integers(2**31)),
            }

        # Create network with fully connected Hebbian links
        self.network = CrossModalNetwork(
            som_configs=som_configs,
            hebbian_eta=cfg["hebbian_eta"],
            hebbian_decay=cfg["hebbian_decay"],
        )

        # Compute SOM representation dimension
        self._som_repr_dim = sum(
            som.n_neurons for som in self.network.soms.values()
        )

        # Update observation space
        new_spaces = {}
        new_spaces["observation"] = self.env.observation_space["observation"]
        new_spaces["som_repr"] = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._som_repr_dim,),
            dtype=np.float32,
        )

        # Propagate goal spaces if present
        if "desired_goal" in self.env.observation_space.spaces:
            new_spaces["desired_goal"] = self.env.observation_space["desired_goal"]
            new_spaces["achieved_goal"] = self.env.observation_space["achieved_goal"]

        self.observation_space = spaces.Dict(new_spaces)

        self._initialized = True

    def _transform_obs(self, obs: dict) -> dict:
        """Transform raw observation through SOM pipeline.

        Args:
            obs: Raw observation dict from MIMo environment.

        Returns:
            Transformed observation dict with SOM representation.
        """
        # Preprocess touch into discriminative/affective channels
        touch_features = self.preprocessor.process(normalize=True)

        # Build SOM inputs
        som_inputs = {
            "tactile_disc": touch_features["discriminative"],
        }
        if "tactile_aff" in self.network.soms:
            som_inputs["tactile_aff"] = touch_features["affective"]
        if "proprio" in self.network.soms:
            som_inputs["proprio"] = obs["observation"]

        # SOM + Hebbian learning (if enabled)
        if self._learning_enabled:
            self.network.learn(som_inputs)

        # Get concatenated SOM representation
        som_repr = self.network.get_representation(som_inputs)

        # Build new observation
        new_obs = {
            "observation": obs["observation"],
            "som_repr": som_repr.astype(np.float32),
        }

        # Propagate goals if present
        if "desired_goal" in obs:
            new_obs["desired_goal"] = obs["desired_goal"]
            new_obs["achieved_goal"] = obs["achieved_goal"]

        return new_obs

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        """Reset environment and initialize SOM if needed."""
        obs, info = self.env.reset(**kwargs)
        self._lazy_init()
        return self._transform_obs(obs), info

    def step(self, action) -> Tuple[dict, float, bool, bool, dict]:
        """Step environment and transform observation through SOM."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_obs = self._transform_obs(obs)

        # Add SOM metrics to info
        info["som_metrics"] = self.network.get_metrics()

        return new_obs, reward, terminated, truncated, info

    def set_learning(self, enabled: bool):
        """Enable or disable SOM/Hebbian learning.

        Useful for evaluation: disable learning to get pure inference.

        Args:
            enabled: Whether to learn during forward passes.
        """
        self._learning_enabled = enabled

    def get_som_state(self) -> dict:
        """Get complete SOM + Hebbian state for checkpointing."""
        if not self._initialized:
            return {}
        return self.network.get_state()

    def set_som_state(self, state: dict):
        """Restore SOM + Hebbian state from checkpoint."""
        if self._initialized:
            self.network.set_state(state)
