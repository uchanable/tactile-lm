"""Developmental trajectory models for tactile maturation.

Models the postnatal development of tactile afferent systems from birth
to ~24 months, including myelination progression, CT afferent maturation,
and sensor density changes due to body surface area growth.

References:
    Palmu, K., et al. (2004). Somatosensory evoked potentials in neonates.
    Fitzgerald, M. (2005). The development of nociceptive circuits. Nat Rev Neurosci.
    Della Longa, L., et al. (2019). Sensitivity to social contingency in infants.
"""

import numpy as np


class DevelopmentalProfile:
    """Models age-dependent tactile afferent maturation.

    Provides scaling factors for conduction velocity, receptor sensitivity,
    and effective sensor density as a function of age (in months, 0-24).

    Attributes:
        age_months: Current developmental age in months (0-24).
    """

    def __init__(self, age_months: float = 18.0):
        """Initialize with a developmental age.

        Args:
            age_months: Age in months. Clamped to [0, 24].
        """
        self.age_months = np.clip(age_months, 0.0, 24.0)

    # ------------------------------------------------------------------
    # Myelination / conduction velocity
    # ------------------------------------------------------------------

    def myelination_factor(self) -> float:
        """Myelination progression factor (0 to 1).

        Myelinated A-beta fibres (SA-I, FA-I, FA-II) have conduction
        velocities that increase with myelination. At birth, conduction
        velocity is roughly 30-40% of adult values and reaches near-adult
        levels by ~24 months.

        CT afferents are unmyelinated (C-fibres) and therefore unaffected
        by myelination, but their central processing pathways still mature.

        Returns:
            float: Fraction of adult conduction velocity [0.3, 1.0].
        """
        # Sigmoid-like maturation curve
        # At birth (~0 months): ~0.3
        # At 12 months: ~0.75
        # At 24 months: ~0.95
        return 0.3 + 0.7 * (1.0 - np.exp(-self.age_months / 8.0))

    def conduction_velocity(self, fibre_type: str = "A_beta") -> float:
        """Estimated conduction velocity in m/s.

        Args:
            fibre_type: "A_beta" for myelinated afferents (SA-I, FA-I, FA-II)
                        or "C" for unmyelinated CT afferents.

        Returns:
            float: Conduction velocity in m/s.
        """
        if fibre_type == "A_beta":
            adult_velocity = 50.0  # m/s, typical adult A-beta
            return adult_velocity * self.myelination_factor()
        elif fibre_type == "C":
            # CT afferents are unmyelinated: ~1 m/s, relatively stable
            return 1.0
        else:
            raise ValueError(f"Unknown fibre type: {fibre_type}")

    # ------------------------------------------------------------------
    # CT afferent maturation
    # ------------------------------------------------------------------

    def ct_maturity(self) -> float:
        """CT afferent functional maturity factor (0 to 1).

        CT afferents are present from birth but their central processing
        (insular cortex integration) matures over the first two years.
        This factor scales the effective CT response.

        Returns:
            float: CT maturity scaling factor [0.4, 1.0].
        """
        # Neonates show CT-optimal responses from birth (Joensson et al. 2018)
        # but cortical processing matures gradually.
        # At birth: ~0.4 (functional but immature central processing)
        # At 24 months: ~0.95
        return 0.4 + 0.6 * (1.0 - np.exp(-self.age_months / 10.0))

    # ------------------------------------------------------------------
    # Myelinated afferent maturation
    # ------------------------------------------------------------------

    def myelinated_maturity(self) -> float:
        """Myelinated afferent (SA-I, FA-I, FA-II) maturity factor.

        Combines myelination progression with receptor maturation.

        Returns:
            float: Overall maturity scaling factor [0.2, 1.0].
        """
        return 0.2 + 0.8 * (1.0 - np.exp(-self.age_months / 7.0))

    # ------------------------------------------------------------------
    # Sensor density scaling
    # ------------------------------------------------------------------

    def density_factor(self) -> float:
        """Relative sensor density factor due to body growth.

        As the body surface area increases with age, the innervation density
        (receptors per unit area) effectively decreases for areas that do
        not add new receptors. However, the total number of sensors stays
        roughly constant in MIMo's discrete model. This factor provides a
        correction for analytical calculations.

        An infant at birth has ~0.25 m^2 BSA; at 24 months ~0.53 m^2.
        Receptor density is inversely proportional to BSA (assuming a fixed
        total receptor count).

        Returns:
            float: Density scaling factor relative to newborn. At birth = 1.0,
                   at 24 months ~ 0.47.
        """
        # Approximate BSA growth (Mosteller formula, simplified)
        # Birth BSA ~ 0.25 m^2, 24-month BSA ~ 0.53 m^2
        bsa_birth = 0.25
        bsa_at_age = bsa_birth + 0.28 * (self.age_months / 24.0)
        return bsa_birth / bsa_at_age

    # ------------------------------------------------------------------
    # Composite scaling per receptor type
    # ------------------------------------------------------------------

    def get_receptor_scale(self, receptor_type: str) -> float:
        """Get the composite developmental scaling factor for a receptor type.

        This combines maturation and density effects into a single multiplier
        that can be applied to the receptor's response.

        Args:
            receptor_type: One of "SA1", "FA1", "FA2", "CT".

        Returns:
            float: A scaling factor in (0, 1] to multiply with the adult response.
        """
        if receptor_type == "CT":
            return self.ct_maturity()
        elif receptor_type in ("SA1", "FA1", "FA2"):
            return self.myelinated_maturity()
        else:
            raise ValueError(f"Unknown receptor type: {receptor_type}")

    def summary(self) -> dict:
        """Return a summary dictionary of all developmental parameters.

        Returns:
            dict: Keys are parameter names, values are current values.
        """
        return {
            "age_months": self.age_months,
            "myelination_factor": self.myelination_factor(),
            "conduction_velocity_A_beta": self.conduction_velocity("A_beta"),
            "conduction_velocity_C": self.conduction_velocity("C"),
            "ct_maturity": self.ct_maturity(),
            "myelinated_maturity": self.myelinated_maturity(),
            "density_factor": self.density_factor(),
            "scale_SA1": self.get_receptor_scale("SA1"),
            "scale_FA1": self.get_receptor_scale("FA1"),
            "scale_FA2": self.get_receptor_scale("FA2"),
            "scale_CT": self.get_receptor_scale("CT"),
        }
