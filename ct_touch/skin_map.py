"""Skin type mapping for MIMo body parts.

Maps each MIMo body name to a skin type (hairy or glabrous). CT afferents
are present only in hairy skin. Glabrous skin (palms, fingertips, soles)
lacks CT afferents but has higher density of SA-I and FA-I receptors.

References:
    Vallbo, A. B., et al. (1999). Unmyelinated afferents constitute a second
    system coding tactile stimuli of the human hairy skin. J Neurophysiol.
"""

from enum import Enum
from typing import Dict


class SkinType(Enum):
    """Skin type classification for mechanoreceptor distribution."""
    HAIRY = "hairy"
    GLABROUS = "glabrous"


# Mapping from MIMo body names to skin type.
# Hairy skin contains CT afferents; glabrous skin does not.
SKIN_TYPE_MAP: Dict[str, SkinType] = {
    # --- Lower extremities ---
    "left_toes": SkinType.GLABROUS,
    "right_toes": SkinType.GLABROUS,
    "left_foot": SkinType.GLABROUS,       # sole is glabrous
    "right_foot": SkinType.GLABROUS,
    "left_lower_leg": SkinType.HAIRY,
    "right_lower_leg": SkinType.HAIRY,
    "left_upper_leg": SkinType.HAIRY,
    "right_upper_leg": SkinType.HAIRY,

    # --- Torso ---
    "hip": SkinType.HAIRY,
    "lower_body": SkinType.HAIRY,
    "upper_body": SkinType.HAIRY,
    "chest": SkinType.HAIRY,

    # --- Head / face ---
    "head": SkinType.HAIRY,
    "left_eye": SkinType.HAIRY,
    "right_eye": SkinType.HAIRY,

    # --- Upper extremities ---
    "left_upper_arm": SkinType.HAIRY,
    "right_upper_arm": SkinType.HAIRY,
    "left_lower_arm": SkinType.HAIRY,
    "right_lower_arm": SkinType.HAIRY,
    "left_hand": SkinType.GLABROUS,       # palm is glabrous
    "right_hand": SkinType.GLABROUS,
    "left_fingers": SkinType.GLABROUS,
    "right_fingers": SkinType.GLABROUS,

    # --- V2 finger bodies (all fingertip/knuckle = glabrous) ---
    "left_ffdistal": SkinType.GLABROUS,
    "left_mfdistal": SkinType.GLABROUS,
    "left_rfdistal": SkinType.GLABROUS,
    "left_lfdistal": SkinType.GLABROUS,
    "left_thdistal": SkinType.GLABROUS,
    "left_ffmiddle": SkinType.GLABROUS,
    "left_mfmiddle": SkinType.GLABROUS,
    "left_rfmiddle": SkinType.GLABROUS,
    "left_lfmiddle": SkinType.GLABROUS,
    "left_thhub": SkinType.GLABROUS,
    "left_ffknuckle": SkinType.GLABROUS,
    "left_mfknuckle": SkinType.GLABROUS,
    "left_rfknuckle": SkinType.GLABROUS,
    "left_lfknuckle": SkinType.GLABROUS,
    "left_thbase": SkinType.GLABROUS,
    "left_lfmetacarpal": SkinType.GLABROUS,

    "right_ffdistal": SkinType.GLABROUS,
    "right_mfdistal": SkinType.GLABROUS,
    "right_rfdistal": SkinType.GLABROUS,
    "right_lfdistal": SkinType.GLABROUS,
    "right_thdistal": SkinType.GLABROUS,
    "right_ffmiddle": SkinType.GLABROUS,
    "right_mfmiddle": SkinType.GLABROUS,
    "right_rfmiddle": SkinType.GLABROUS,
    "right_lfmiddle": SkinType.GLABROUS,
    "right_thhub": SkinType.GLABROUS,
    "right_ffknuckle": SkinType.GLABROUS,
    "right_mfknuckle": SkinType.GLABROUS,
    "right_rfknuckle": SkinType.GLABROUS,
    "right_lfknuckle": SkinType.GLABROUS,
    "right_thbase": SkinType.GLABROUS,
    "right_lfmetacarpal": SkinType.GLABROUS,

    # --- V2 toe bodies ---
    "left_big_toe": SkinType.GLABROUS,
    "right_big_toe": SkinType.GLABROUS,
}


# Receptor density parameters by skin type (receptors per cm^2).
# Based on Johansson & Vallbo (1979), Vallbo et al. (1999).
RECEPTOR_DENSITY: Dict[str, Dict[str, float]] = {
    "hairy": {
        "SA1": 0.7,     # Merkel cells - lower density on hairy skin
        "FA1": 0.6,     # Meissner corpuscles - sparse on hairy skin
        "FA2": 0.02,    # Pacinian - deep subcutaneous, sparse everywhere
        "CT": 1.6,      # CT afferents - only on hairy skin
    },
    "glabrous": {
        "SA1": 7.0,     # Merkel cells - high density on fingertips
        "FA1": 8.5,     # Meissner corpuscles - high density
        "FA2": 0.02,    # Pacinian - same sparse density
        "CT": 0.0,      # No CT afferents on glabrous skin
    },
}


def get_skin_type(body_name: str) -> SkinType:
    """Return the skin type for a MIMo body part.

    Args:
        body_name: The name of the MIMo body part.

    Returns:
        SkinType: HAIRY or GLABROUS. Defaults to HAIRY for unknown parts.
    """
    return SKIN_TYPE_MAP.get(body_name, SkinType.HAIRY)


def has_ct_afferents(body_name: str) -> bool:
    """Check whether a body part has CT afferents.

    Args:
        body_name: The name of the MIMo body part.

    Returns:
        bool: True if the body part has hairy skin (and thus CT afferents).
    """
    return get_skin_type(body_name) == SkinType.HAIRY
