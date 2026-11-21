"""Power-asset condition recognition reproduction package."""

CLASS_ORDER = (
    "glass-insulator::good", "glass-insulator::missing-cap",
    "lightning-rod-suspension::good", "lightning-rod-suspension::rust",
    "polymer-insulator-upper-shackle::good", "polymer-insulator-upper-shackle::rust",
    "vari-grip::bird-nest", "vari-grip::good", "vari-grip::rust",
    "yoke-suspension::good", "yoke-suspension::rust",
)
OBSERVATION_KEYS = (
    "equipment", "visible_components", "structural_state", "surface_state",
    "missing_or_broken_parts", "abnormal_visual_cues", "spatial_relations",
    "uncertain_observations",
)
