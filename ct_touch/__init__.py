"""CT afferent augmented touch module for MIMo.

This package extends MIMo's TrimeshTouch with CT (C-tactile) afferent
mechanoreceptor models, multi-receptor output, skin-type mapping, and
developmental trajectories.

Import convenience aliases (lazy to avoid circular imports):

    from ct_touch import CTAugmentedTouch
    from ct_touch import SkinType, get_skin_type, SKIN_TYPE_MAP
    from ct_touch import DevelopmentalProfile
"""


def __getattr__(name):
    """Lazy imports to avoid circular dependency with mimoTouch."""
    if name == "CTAugmentedTouch":
        from ct_touch.ct_augmented_touch import CTAugmentedTouch
        return CTAugmentedTouch
    if name in ("SKIN_TYPE_MAP", "SkinType", "get_skin_type"):
        from ct_touch import skin_map
        return getattr(skin_map, name)
    if name == "DevelopmentalProfile":
        from ct_touch.developmental import DevelopmentalProfile
        return DevelopmentalProfile
    raise AttributeError(f"module 'ct_touch' has no attribute {name!r}")
