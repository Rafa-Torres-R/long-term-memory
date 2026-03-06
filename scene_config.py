# Purpose: Stores scene knowledge
# pythonOBJECTS = {
#     "024_bowl-0": {...},  # Object positions
#     "024_bowl-3": {"initial_location": "drawer", ...}
# }

# ARTICULATIONS = {
#     "drawer": {
#         "articulation_type": "kitchen_counter",
#         "articulation_handle_link_idx": 7,
#         ...
#     }
# }

# FRIENDLY_NAME_TO_OBJ_ID = {
#     "bowl_0": "024_bowl-0",
#     "bowl_3": "024_bowl-3"
# }

"""
Scene Configuration: Stores all articulation details and object positions
"""

# Drawer coordinates (when open)
DRAWER_CENTER_X = -1.7
DRAWER_CENTER_Y = 0.04
DRAWER_CENTER_Z = 0.52

# Object placement offsets (to avoid collisions when placing multiple objects)
OBJECT_OFFSETS = {
    0: [0.0, 0.0],
    1: [0.01, 0.0],
    2: [-0.01, 0.0],
    3: [0.0, 0.01],
    4: [0.0, -0.01],
}

# Articulation configurations
ARTICULATIONS = {
    "drawer": {
        "articulation_type": "kitchen_counter",
        "articulation_id": "kitchen_counter-0",
        "articulation_handle_link_idx": 7,
        "articulation_handle_active_joint_idx": 5,
        "articulation_relative_handle_pos": [0.26, 0.0, 0],
        "placement_coords": {
            "center": [DRAWER_CENTER_X, DRAWER_CENTER_Y, DRAWER_CENTER_Z],
        }
    },
    "fridge": {
        "articulation_type": "fridge",
        "articulation_id": "fridge-0",
        "articulation_handle_link_idx": 2,
        "articulation_handle_active_joint_idx": 0,
        "articulation_relative_handle_pos": [0.09, -0.66, 0.2],
    }
}

# Available objects - BASE NAMES (ManiSkill adds env-0_ prefix automatically)
OBJECTS = {
    "013_apple-0": {
        "type": "apple",
        "name": "013_apple",
        "friendly_name": "apple",
        "initial_location": "counter",
        "initial_position": None,
    },
    "024_bowl-0": {
        "type": "bowl",
        "name": "024_bowl",
        "friendly_name": "bowl_0",
        "initial_location": "counter",
        "initial_position": [-2.45844, 0.88718, 0.64234],
    },
    "024_bowl-1": {
        "type": "bowl", 
        "name": "024_bowl",
        "friendly_name": "bowl_1",
        "initial_location": "counter",
        "initial_position": [-1.80886, 0.96162, 3.3442],
    },
    "024_bowl-3": {
        "type": "bowl",
        "name": "024_bowl",
        "friendly_name": "bowl_3",
        "initial_location": "drawer",
        "initial_position": [-2.15972, 0.5151, -0.04666],
    }
}

# Map friendly names to BASE object IDs (without env-0_ prefix)
FRIENDLY_NAME_TO_OBJ_ID = {
    "apple": "013_apple-0",
    "bowl_0": "024_bowl-0",
    "bowl_1": "024_bowl-1",
    "bowl_3": "024_bowl-3",
}

# Valid object IDs
VALID_OBJECT_IDS = list(OBJECTS.keys())
