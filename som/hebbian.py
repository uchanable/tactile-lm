"""Hebbian cross-modal binding between SOMs.

Implements Hebbian associative learning between pairs of Self-Organizing
Maps, enabling cross-modal representation binding (e.g., tactile-
discriminative <-> tactile-affective <-> proprioceptive).

Reference:
    Hebb, D. O. (1949). The Organization of Behavior. Wiley.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from som.core import SelfOrganizingMap


class HebbianLink:
    """Bidirectional Hebbian association between two SOMs.

    Maintains a weight matrix W where W[i,j] represents the association
    strength between neuron i in SOM_a and neuron j in SOM_b. Learning
    follows Hebbian coincidence: co-activated neurons strengthen their
    connection.

    Attributes:
        W: Weight matrix of shape (n_a, n_b).
        som_a_name: Name identifier for the first SOM.
        som_b_name: Name identifier for the second SOM.
    """

    def __init__(
        self,
        n_a: int,
        n_b: int,
        som_a_name: str = "a",
        som_b_name: str = "b",
        eta: float = 0.01,
        decay: float = 0.001,
        threshold: float = 0.05,
    ):
        """Initialize Hebbian link.

        Args:
            n_a: Number of neurons in SOM_a.
            n_b: Number of neurons in SOM_b.
            som_a_name: Name for SOM_a (for logging).
            som_b_name: Name for SOM_b (for logging).
            eta: Hebbian learning rate.
            decay: Weight decay rate per step (prevents saturation).
            threshold: Minimum activation to trigger learning.
        """
        self.n_a = n_a
        self.n_b = n_b
        self.som_a_name = som_a_name
        self.som_b_name = som_b_name
        self.eta = eta
        self.decay = decay
        self.threshold = threshold

        self.W = np.zeros((n_a, n_b), dtype=np.float64)
        self._step = 0

    def update(self, act_a: np.ndarray, act_b: np.ndarray):
        """Hebbian learning step.

        Updates W based on co-activation of the two SOMs:
            ΔW[i,j] = η * a_i * b_j  (for activations above threshold)
            W *= (1 - decay)          (weight decay)

        Args:
            act_a: Activation map from SOM_a, shape (n_a,).
            act_b: Activation map from SOM_b, shape (n_b,).
        """
        # Threshold: only learn from sufficiently active neurons
        mask_a = act_a > self.threshold
        mask_b = act_b > self.threshold

        if mask_a.any() and mask_b.any():
            # Outer product of thresholded activations
            a_masked = act_a * mask_a
            b_masked = act_b * mask_b
            self.W += self.eta * np.outer(a_masked, b_masked)

        # Weight decay
        self.W *= (1.0 - self.decay)

        # Clip to prevent runaway
        np.clip(self.W, 0.0, 1.0, out=self.W)

        self._step += 1

    def predict_b(self, act_a: np.ndarray) -> np.ndarray:
        """Predict SOM_b activation from SOM_a activation.

        Args:
            act_a: Activation map from SOM_a, shape (n_a,).

        Returns:
            np.ndarray: Predicted activation for SOM_b, shape (n_b,).
        """
        pred = act_a @ self.W
        norm = pred.max()
        if norm > 0:
            pred /= norm
        return pred

    def predict_a(self, act_b: np.ndarray) -> np.ndarray:
        """Predict SOM_a activation from SOM_b activation.

        Args:
            act_b: Activation map from SOM_b, shape (n_b,).

        Returns:
            np.ndarray: Predicted activation for SOM_a, shape (n_a,).
        """
        pred = self.W @ act_b
        norm = pred.max()
        if norm > 0:
            pred /= norm
        return pred

    def binding_strength(self) -> float:
        """Overall binding strength (mean non-zero weight).

        Higher values indicate stronger cross-modal association.

        Returns:
            float: Mean weight value.
        """
        return float(np.mean(self.W))

    def specificity(self) -> float:
        """How selective the binding is (entropy-based).

        Low entropy = highly specific (few strong connections).
        High entropy = diffuse (many weak connections).
        Returns normalized specificity in [0, 1] where 1 = maximally specific.

        Returns:
            float: Specificity score.
        """
        w_sum = self.W.sum()
        if w_sum < 1e-10:
            return 0.0
        p = self.W.ravel() / w_sum
        p = p[p > 1e-10]  # Filter zeros
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(self.n_a * self.n_b)
        return 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0

    def get_state(self) -> dict:
        """Serialize state."""
        return {
            "W": self.W.copy(),
            "step": self._step,
            "eta": self.eta,
            "decay": self.decay,
        }

    def set_state(self, state: dict):
        """Restore state."""
        self.W = state["W"].copy()
        self._step = state["step"]


class CrossModalNetwork:
    """Network of SOMs connected by Hebbian links.

    Manages multiple SOMs and their pairwise Hebbian connections.
    Default configuration for CT-Touch:
    - tactile_disc: Discriminative touch SOM
    - tactile_aff: Affective touch (CT) SOM
    - proprio: Proprioceptive SOM

    Attributes:
        soms: Dict mapping name -> SelfOrganizingMap.
        links: Dict mapping (name_a, name_b) -> HebbianLink.
    """

    def __init__(
        self,
        som_configs: Dict[str, dict],
        link_pairs: Optional[List[Tuple[str, str]]] = None,
        hebbian_eta: float = 0.01,
        hebbian_decay: float = 0.001,
    ):
        """Initialize cross-modal network.

        Args:
            som_configs: Dict mapping SOM name -> kwargs for SelfOrganizingMap.
                Example: {"tactile_disc": {"grid_size": (15,15), "input_dim": 96}}
            link_pairs: List of (name_a, name_b) pairs to connect.
                If None, creates fully connected links.
            hebbian_eta: Learning rate for all Hebbian links.
            hebbian_decay: Weight decay for all Hebbian links.
        """
        # Create SOMs
        self.soms: Dict[str, SelfOrganizingMap] = {}
        for name, config in som_configs.items():
            self.soms[name] = SelfOrganizingMap(**config)

        # Create Hebbian links
        if link_pairs is None:
            # Fully connected
            names = list(self.soms.keys())
            link_pairs = [
                (names[i], names[j])
                for i in range(len(names))
                for j in range(i + 1, len(names))
            ]

        self.links: Dict[Tuple[str, str], HebbianLink] = {}
        for name_a, name_b in link_pairs:
            n_a = self.soms[name_a].n_neurons
            n_b = self.soms[name_b].n_neurons
            self.links[(name_a, name_b)] = HebbianLink(
                n_a, n_b,
                som_a_name=name_a,
                som_b_name=name_b,
                eta=hebbian_eta,
                decay=hebbian_decay,
            )

        self._activations: Dict[str, np.ndarray] = {}

    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward pass: compute BMUs and activation maps for all SOMs.

        Args:
            inputs: Dict mapping SOM name -> input vector.
                Missing keys are skipped.

        Returns:
            Dict mapping SOM name -> activation map.
        """
        self._activations = {}
        for name, x in inputs.items():
            if name in self.soms:
                self._activations[name] = self.soms[name].get_activation_map(x)
        return self._activations

    def learn(self, inputs: Dict[str, np.ndarray]):
        """Combined SOM + Hebbian learning step.

        1. Update each SOM with its input (Kohonen learning).
        2. Compute activation maps.
        3. Update Hebbian links between co-active SOMs.

        Args:
            inputs: Dict mapping SOM name -> input vector.
        """
        # Step 1: SOM learning
        for name, x in inputs.items():
            if name in self.soms:
                self.soms[name].update(x)

        # Step 2: Compute activations
        activations = self.forward(inputs)

        # Step 3: Hebbian learning on all links where both SOMs have input
        for (name_a, name_b), link in self.links.items():
            if name_a in activations and name_b in activations:
                link.update(activations[name_a], activations[name_b])

    def get_representation(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get concatenated SOM activation maps as a flat vector.

        This is the observation vector that can be fed to a downstream
        RL agent (e.g., PPO).

        Args:
            inputs: Dict mapping SOM name -> input vector.

        Returns:
            np.ndarray: Concatenated activation maps from all SOMs.
        """
        activations = self.forward(inputs)
        parts = []
        for name in sorted(self.soms.keys()):
            if name in activations:
                parts.append(activations[name])
            else:
                parts.append(np.zeros(self.soms[name].n_neurons))
        return np.concatenate(parts)

    def get_metrics(self) -> dict:
        """Collect all quality and binding metrics.

        Returns:
            dict: SOM quality metrics and Hebbian binding strengths.
        """
        metrics = {}

        # SOM metrics (require data, so just report step counts)
        for name, som in self.soms.items():
            metrics[f"som_{name}_step"] = som._step

        # Hebbian metrics
        for (name_a, name_b), link in self.links.items():
            key = f"hebbian_{name_a}_{name_b}"
            metrics[f"{key}_binding"] = link.binding_strength()
            metrics[f"{key}_specificity"] = link.specificity()
            metrics[f"{key}_step"] = link._step

        return metrics

    def cross_modal_prediction_accuracy(
        self,
        test_inputs: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Evaluate cross-modal prediction accuracy.

        For each Hebbian link, predict SOM_b's BMU from SOM_a's activation
        and check if it matches the actual BMU.

        Args:
            test_inputs: Dict mapping SOM name -> input matrix (n_samples, dim).

        Returns:
            Dict mapping link_name -> accuracy (fraction correct).
        """
        # Get number of samples from first available input
        n_samples = next(iter(test_inputs.values())).shape[0]
        accuracies = {}

        for (name_a, name_b), link in self.links.items():
            if name_a not in test_inputs or name_b not in test_inputs:
                continue

            correct = 0
            for i in range(n_samples):
                x_a = test_inputs[name_a][i]
                x_b = test_inputs[name_b][i]

                # Actual BMU in SOM_b
                actual_bmu = self.soms[name_b].find_bmu(x_b)

                # Predicted BMU from SOM_a -> Hebbian -> SOM_b
                act_a = self.soms[name_a].get_activation_map(x_a)
                pred_act_b = link.predict_b(act_a)
                pred_bmu = int(np.argmax(pred_act_b))

                if pred_bmu == actual_bmu:
                    correct += 1

            accuracies[f"{name_a}->{name_b}"] = correct / n_samples

        return accuracies

    def get_state(self) -> dict:
        """Serialize entire network state."""
        return {
            "soms": {name: som.get_state() for name, som in self.soms.items()},
            "links": {
                f"{a}_{b}": link.get_state()
                for (a, b), link in self.links.items()
            },
        }

    def set_state(self, state: dict):
        """Restore network state."""
        for name, som_state in state["soms"].items():
            if name in self.soms:
                self.soms[name].set_state(som_state)
        for key, link_state in state["links"].items():
            for (a, b), link in self.links.items():
                if f"{a}_{b}" == key:
                    link.set_state(link_state)
                    break
