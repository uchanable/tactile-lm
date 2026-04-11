"""Self-Organizing Map (Kohonen, 1982) implementation.

A 2D rectangular SOM with exponentially decaying learning rate and
neighborhood radius. Designed for tactile representation learning
in the MIMo infant simulation.

Reference:
    Kohonen, T. (2001). Self-Organizing Maps, 3rd ed. Springer.
"""

import numpy as np
from typing import Tuple, Optional


class SelfOrganizingMap:
    """2D Self-Organizing Map with Kohonen learning rule.

    Attributes:
        grid_h: Number of rows in the SOM grid.
        grid_w: Number of columns in the SOM grid.
        input_dim: Dimensionality of input vectors.
        weights: Weight matrix of shape (grid_h * grid_w, input_dim).
        n_neurons: Total number of neurons (grid_h * grid_w).
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        input_dim: int = 96,
        initial_lr: float = 0.5,
        final_lr: float = 0.01,
        initial_sigma: float = None,
        final_sigma: float = 0.5,
        decay_steps: int = 100_000,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the SOM.

        Args:
            grid_size: (height, width) of the 2D neuron grid.
            input_dim: Dimensionality of input vectors.
            initial_lr: Initial learning rate.
            final_lr: Final learning rate after decay.
            initial_sigma: Initial neighborhood radius. Defaults to max(grid_size)/2.
            final_sigma: Final neighborhood radius.
            decay_steps: Number of steps for exponential decay.
            rng: NumPy random generator for reproducibility.
        """
        self.grid_h, self.grid_w = grid_size
        self.input_dim = input_dim
        self.n_neurons = self.grid_h * self.grid_w

        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.initial_sigma = initial_sigma or max(grid_size) / 2.0
        self.final_sigma = final_sigma
        self.decay_steps = decay_steps

        self._rng = rng or np.random.default_rng()
        self._step = 0

        # Initialize weights uniformly in [0, 1]
        self.weights = self._rng.uniform(0, 1, (self.n_neurons, input_dim))

        # Precompute grid coordinates for neighborhood calculation
        rows, cols = np.divmod(np.arange(self.n_neurons), self.grid_w)
        self._grid_coords = np.column_stack([rows, cols]).astype(np.float64)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _get_lr(self) -> float:
        """Current learning rate (exponential decay)."""
        t = min(self._step / self.decay_steps, 1.0)
        return self.initial_lr * (self.final_lr / self.initial_lr) ** t

    def _get_sigma(self) -> float:
        """Current neighborhood radius (exponential decay)."""
        t = min(self._step / self.decay_steps, 1.0)
        return self.initial_sigma * (self.final_sigma / self.initial_sigma) ** t

    def _distances_to_input(self, x: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distance from each neuron to input.

        Args:
            x: Input vector of shape (input_dim,).

        Returns:
            np.ndarray: Distances of shape (n_neurons,).
        """
        diff = self.weights - x[np.newaxis, :]
        return np.sum(diff * diff, axis=1)

    def find_bmu(self, x: np.ndarray) -> int:
        """Find the Best Matching Unit (BMU) for input x.

        Args:
            x: Input vector of shape (input_dim,).

        Returns:
            int: Index of the BMU in the flattened grid.
        """
        distances = self._distances_to_input(x)
        return int(np.argmin(distances))

    def get_bmu_position(self, x: np.ndarray) -> Tuple[int, int]:
        """Get 2D grid position of the BMU.

        Args:
            x: Input vector of shape (input_dim,).

        Returns:
            Tuple[int, int]: (row, col) position in the grid.
        """
        bmu = self.find_bmu(x)
        return int(bmu // self.grid_w), int(bmu % self.grid_w)

    def get_activation_map(self, x: np.ndarray) -> np.ndarray:
        """Compute activation map (inverse distance, normalized).

        Each neuron's activation is exp(-dist / (2*sigma^2)), representing
        how strongly it responds to the input.

        Args:
            x: Input vector of shape (input_dim,).

        Returns:
            np.ndarray: Activation map of shape (n_neurons,), values in [0, 1].
        """
        distances = self._distances_to_input(x)
        sigma = self._get_sigma()
        activations = np.exp(-distances / (2.0 * sigma * sigma + 1e-8))
        return activations

    def update(self, x: np.ndarray) -> int:
        """Perform one Kohonen learning step.

        Args:
            x: Input vector of shape (input_dim,).

        Returns:
            int: Index of the BMU.
        """
        bmu = self.find_bmu(x)
        lr = self._get_lr()
        sigma = self._get_sigma()

        # Neighborhood function: Gaussian on grid distance
        bmu_coord = self._grid_coords[bmu]
        grid_dists_sq = np.sum(
            (self._grid_coords - bmu_coord[np.newaxis, :]) ** 2, axis=1
        )
        h = np.exp(-grid_dists_sq / (2.0 * sigma * sigma + 1e-8))

        # Weight update: Δw = lr * h * (x - w)
        delta = x[np.newaxis, :] - self.weights
        self.weights += lr * h[:, np.newaxis] * delta

        self._step += 1
        return bmu

    def batch_update(self, X: np.ndarray):
        """Update SOM with a batch of input vectors.

        Args:
            X: Input matrix of shape (n_samples, input_dim).
        """
        for x in X:
            self.update(x)

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def quantization_error(self, X: np.ndarray) -> float:
        """Mean distance from each input to its BMU.

        Lower is better — indicates how well the SOM represents the data.

        Args:
            X: Input matrix of shape (n_samples, input_dim).

        Returns:
            float: Mean quantization error.
        """
        total = 0.0
        for x in X:
            bmu = self.find_bmu(x)
            total += np.sqrt(np.sum((x - self.weights[bmu]) ** 2))
        return total / len(X)

    def topographic_error(self, X: np.ndarray) -> float:
        """Proportion of inputs whose 1st and 2nd BMU are not grid neighbors.

        Lower is better — indicates how well the map preserves topology.

        Args:
            X: Input matrix of shape (n_samples, input_dim).

        Returns:
            float: Topographic error in [0, 1].
        """
        errors = 0
        for x in X:
            distances = self._distances_to_input(x)
            sorted_idx = np.argsort(distances)
            bmu1, bmu2 = sorted_idx[0], sorted_idx[1]

            # Check if bmu1 and bmu2 are grid neighbors (4-connectivity)
            r1, c1 = bmu1 // self.grid_w, bmu1 % self.grid_w
            r2, c2 = bmu2 // self.grid_w, bmu2 % self.grid_w
            grid_dist = abs(r1 - r2) + abs(c1 - c2)
            if grid_dist > 1:
                errors += 1

        return errors / len(X)

    def u_matrix(self) -> np.ndarray:
        """Compute the U-matrix (unified distance matrix).

        Each cell contains the mean distance to its grid neighbors'
        weight vectors. High values indicate cluster boundaries.

        Returns:
            np.ndarray: U-matrix of shape (grid_h, grid_w).
        """
        u_mat = np.zeros((self.grid_h, self.grid_w))
        weights_2d = self.weights.reshape(self.grid_h, self.grid_w, self.input_dim)

        for r in range(self.grid_h):
            for c in range(self.grid_w):
                neighbors = []
                if r > 0:
                    neighbors.append(weights_2d[r - 1, c])
                if r < self.grid_h - 1:
                    neighbors.append(weights_2d[r + 1, c])
                if c > 0:
                    neighbors.append(weights_2d[r, c - 1])
                if c < self.grid_w - 1:
                    neighbors.append(weights_2d[r, c + 1])

                dists = [np.linalg.norm(weights_2d[r, c] - n) for n in neighbors]
                u_mat[r, c] = np.mean(dists)

        return u_mat

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def init_from_data(self, X: np.ndarray):
        """Initialize weights using PCA of the data.

        Places the SOM grid along the first two principal components,
        which typically leads to faster convergence than random init.

        Args:
            X: Input matrix of shape (n_samples, input_dim).
        """
        mean = X.mean(axis=0)
        X_centered = X - mean

        # SVD for top 2 components
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        pc1 = Vt[0] * S[0] / np.sqrt(len(X))
        pc2 = Vt[1] * S[1] / np.sqrt(len(X)) if len(Vt) > 1 else np.zeros_like(pc1)

        # Span the grid along PC1 and PC2
        for i in range(self.n_neurons):
            r = (i // self.grid_w) / max(self.grid_h - 1, 1) * 2 - 1  # [-1, 1]
            c = (i % self.grid_w) / max(self.grid_w - 1, 1) * 2 - 1   # [-1, 1]
            self.weights[i] = mean + r * pc1 + c * pc2

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Get serializable state."""
        return {
            "grid_size": (self.grid_h, self.grid_w),
            "input_dim": self.input_dim,
            "weights": self.weights.copy(),
            "step": self._step,
            "initial_lr": self.initial_lr,
            "final_lr": self.final_lr,
            "initial_sigma": self.initial_sigma,
            "final_sigma": self.final_sigma,
            "decay_steps": self.decay_steps,
        }

    def set_state(self, state: dict):
        """Restore from serialized state."""
        self.weights = state["weights"].copy()
        self._step = state["step"]
