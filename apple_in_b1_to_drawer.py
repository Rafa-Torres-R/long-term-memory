import json

# Drawer center coordinates
DRAWER_CENTER_X = -1.7
DRAWER_CENTER_Y = 0.04
DRAWER_CENTER_Z = 0.52

# Bowl_1 original position coordinates (where apple starts)
BOWL_1_X = -1.7
BOWL_1_Y = -3.3442 
BOWL_1_Z = 0.96162

# Task: Pick apple from bowl_1 position and place in drawer
task = {
  "plans": [{
    "build_config_name": "v3_sc1_staging_04.scene_instance.json",
    "init_config_name": "/home/fri/ManiSkill-HAB/test_rafa/changing_objects/apple_at_b1.json",
    "subtasks": [
      # Step 1: Navigate to drawer
      {
        "type": "navigate",
        "uid": "nav_to_drawer"
      },
      # Step 2: Open drawer
      {
        "type": "open",
        "uid": "open_drawer",
        "articulation_type": "kitchen_counter",
        "articulation_id": "kitchen_counter-0",
        "articulation_handle_link_idx": 7,
        "articulation_handle_active_joint_idx": 5,
        "articulation_relative_handle_pos": [0.26, 0.0, 0],
        "obj_id": "013_apple-0"
      },
      
      # Step 3: Navigate to apple at bowl_1's position
      {
        "type": "navigate",
        "uid": "nav_to_apple_at_bowl_1"
      },
      # Step 4: Pick apple from counter (at bowl_1 position)
      {
        "type": "pick",
        "uid": "pick_apple_from_bowl_1_spot",
        "obj_id": "013_apple-0",
        "articulation_config": None  # On counter, not in furniture
      },
      
      # Step 5: Navigate to open drawer
      {
        "type": "navigate",
        "uid": "nav_to_open_drawer"
      },
      # Step 6: Place apple in drawer
      {
        "type": "place",
        "uid": "place_apple_in_drawer",
        "obj_id": "013_apple-0",
        "goal_pos": [DRAWER_CENTER_X, DRAWER_CENTER_Y, DRAWER_CENTER_Z],
        "goal_rectangle_corners": [
          [DRAWER_CENTER_X-0.0015, DRAWER_CENTER_Y+0.001, DRAWER_CENTER_Z], 
          [DRAWER_CENTER_X+0.0015, DRAWER_CENTER_Y-0.001, DRAWER_CENTER_Z], 
          [DRAWER_CENTER_X-0.0015, DRAWER_CENTER_Y+0.001, DRAWER_CENTER_Z], 
          [DRAWER_CENTER_X+0.0015, DRAWER_CENTER_Y-0.001, DRAWER_CENTER_Z]
        ],
        "validate_goal_rectangle_corners": False,
        "articulation_config": {
          "articulation_type": "kitchen_counter",
          "articulation_id": "kitchen_counter-0",
          "articulation_handle_link_idx": 7,
          "articulation_handle_active_joint_idx": 5
        }
      },
      
      # Step 7: Navigate away from drawer
      {
        "type": "navigate",
        "uid": "nav_away_from_drawer"
      },
      # Step 8: Close drawer
      {
        "type": "close",
        "uid": "close_drawer",
        "articulation_type": "kitchen_counter",
        "articulation_id": "kitchen_counter-0",
        "articulation_handle_link_idx": 7,
        "articulation_handle_active_joint_idx": 5,
        "articulation_relative_handle_pos": [0.26, 0.0, 0],
        "remove_obj_id": None
      }
    ]
  }],
  "dataset": "ReplicaCADRearrangeDataset"
}

# Save task
output_paths = [
  "/home/fri/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/sequential/train/custom_apple_drawer.json",
  "/home/fri/ManiSkill-HAB/test_rafa/custom_task_plans/finals/apple_bowl1_to_drawer.json"
]

for path in output_paths:
  with open(path, "w") as f:
    json.dump(task, f, indent=2)
  print(f"✓ Saved to: {path}")

print("\n" + "=" * 70)
print("TASK CREATED: Pick Apple from Bowl_1 Position → Place in Drawer")
print("=" * 70)
print("\nSequence:")
print("  1. Navigate to drawer → Open drawer")
print("  2. Navigate to apple at bowl_1 spot → Pick apple")
print("  3. Navigate to drawer → Place apple in drawer")
print("  4. Navigate away → Close drawer")
print(f"\nApple starts at: [x={BOWL_1_X}, y={BOWL_1_Y}, z={BOWL_1_Z}]")
print(f"Apple ends at:   [x={DRAWER_CENTER_X}, y={DRAWER_CENTER_Y}, z={DRAWER_CENTER_Z}]")
print("\n8 subtasks total (~2-3 minutes)")
print("=" * 70)
