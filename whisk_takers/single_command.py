"""
Single Command Execution
- Execute ONE natural language command
- Get video output
- Exit

Usage:
    cd ~/ManiSkill-HAB
    python whisk_takers/single_command.py "pick up bowl_3"
"""
import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
MSHAB_ROOT            = Path("/home/fri/ManiSkill-HAB")
WHISK_TAKERS_DIR      = MSHAB_ROOT / "whisk_takers"
INTERACTIVE_ROBOT_DIR = MSHAB_ROOT / "test_rafa/interactive_robot"

os.chdir(str(MSHAB_ROOT))
sys.path.insert(0, str(MSHAB_ROOT))
sys.path.insert(0, str(WHISK_TAKERS_DIR))
sys.path.insert(0, str(INTERACTIVE_ROBOT_DIR))

os.environ["SAPIEN_NO_DISPLAY"] = "1"

# -----------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------
from mshab.envs.make import EnvConfig, make_env
from mshab.envs import SequentialTaskEnv
from mshab.envs.planner import (
    PickSubtask, PlaceSubtask, NavigateSubtask, OpenSubtask, CloseSubtask
)
from mshab.agents.sac.agent import Agent as SACAgent
from gymnasium import spaces
from mshab.agents.ppo.agent import Agent as PPOAgent
from mshab.utils.array import to_tensor, recursive_slice
from mshab.utils.config import parse_cfg

# ⭐ Import from whisk_takers
from llm_command_parser_v1 import QwenCommandParser
from subtask_enricher import SubtaskEnricher

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
SEED          = 0
ENV_ID        = "SequentialTask-v0"
NUM_ENVS      = 1
MAX_STEPS     = 1500
FRAME_STACK   = 3
SCENE_BUILDER = "ReplicaCADSetTableTrain"
BUILD_CONFIG  = "v3_sc1_staging_04.scene_instance.json"
INIT_CONFIG   = str("/home/fri/ManiSkill-HAB/test_rafa/changing_objects/original_episode_og.json")
DATASET       = "ReplicaCADRearrangeDataset"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR      = MSHAB_ROOT / "mshab_checkpoints/rl/set_table"

OUTPUT_DIR = WHISK_TAKERS_DIR / "single_command_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBTASK_TYPE_IDS = {"pick": 0, "place": 1, "navigate": 2, "open": 3, "close": 4}

SET_TABLE_TARGS = {
    "pick":     ["013_apple", "024_bowl", "all"],
    "place":    ["013_apple", "024_bowl", "all"],
    "navigate": ["all"],
    "open":     ["fridge", "kitchen_counter"],
    "close":    ["fridge", "kitchen_counter"],
}

ALL_TARG_NAMES = set()
for targs in SET_TABLE_TARGS.values():
    ALL_TARG_NAMES.update(targs)

# -----------------------------------------------------------------------
# [Keep all the helper functions exactly as before]
# -----------------------------------------------------------------------
def create_env(task_file: Path, video_path: Path):
    env_cfg = EnvConfig(
        env_id=ENV_ID,
        num_envs=NUM_ENVS,
        max_episode_steps=MAX_STEPS,
        make_env=True,
        obs_mode="depth",
        render_mode="all",
        shader_dir="minimal",
        sim_backend="gpu",
        continuous_task=True,
        cat_state=True,
        cat_pixels=False,
        frame_stack=FRAME_STACK,
        stack=None,
        stationary_base=False,
        stationary_torso=False,
        stationary_head=True,
        task_plan_fp=str(task_file),
        record_video=True,
        save_video_freq=1,
        info_on_video=False,
        env_kwargs={
            "require_build_configs_repeated_equally_across_envs": False,
            "add_event_tracker_info": True,
            "task_cfgs": {"navigate": {"ignore_arm_checkers": True}},
            "scene_builder_cls": SCENE_BUILDER,
        },
    )
    return make_env(env_cfg, video_path=str(video_path))

def load_policies(env, dummy_obs):
    uenv: SequentialTaskEnv = env.unwrapped
    obs_space = uenv.single_observation_space
    act_space = uenv.single_action_space

    model_pixel_obs_space = {}
    for k, space in obs_space["pixels"].items():
        shape, low, high, dtype = space.shape, space.low, space.high, space.dtype
        if len(shape) == 4:
            shape = (shape[0] * shape[1], shape[-2], shape[-1])
            low  = low.reshape((-1, *low.shape[-2:]))
            high = high.reshape((-1, *high.shape[-2:]))
        model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
    model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)
    state_obs_shape = obs_space["state"].shape

    policies = {}
    for subtask_name, targ_names in SET_TABLE_TARGS.items():
        policies[subtask_name] = {}
        for targ_name in targ_names:
            ckpt_path = CKPT_DIR / subtask_name / targ_name / "policy.pt"
            cfg_path  = CKPT_DIR / subtask_name / targ_name / "config.yml"

            if not ckpt_path.exists():
                continue

            algo_name = parse_cfg(default_cfg_path=str(cfg_path)).algo.name

            if algo_name == "sac":
                policy = SACAgent(
                    pixels_obs_space=model_pixel_obs_space,
                    state_obs_shape=state_obs_shape,
                    action_shape=act_space.shape,
                    actor_hidden_dims=[256, 256, 256],
                    critic_hidden_dims=[256, 256, 256],
                    critic_layer_norm=True,
                    critic_dropout=None,
                    encoder_pixels_feature_dim=50,
                    encoder_state_feature_dim=50,
                    cnn_features=[32, 64, 128, 256],
                    cnn_filters=[3, 3, 3, 3],
                    cnn_strides=[2, 2, 2, 2],
                    cnn_padding="valid",
                    log_std_min=-20,
                    log_std_max=2,
                    device=DEVICE,
                )
                policy.eval()
                policy.load_state_dict(
                    torch.load(str(ckpt_path), map_location=DEVICE)["agent"]
                )
                policy.to(DEVICE)
                act_fn = (
                    lambda p: lambda obs: p.actor(
                        obs["pixels"], obs["state"],
                        compute_pi=False, compute_log_pi=False
                    )[0]
                )(policy)

            elif algo_name == "ppo":
                dummy_obs_t = to_tensor(dummy_obs, device=DEVICE, dtype="float")
                policy = PPOAgent(dummy_obs_t, act_space.shape)
                policy.eval()
                policy.load_state_dict(
                    torch.load(str(ckpt_path), map_location=DEVICE)["agent"]
                )
                policy.to(DEVICE)
                act_fn = (
                    lambda p: lambda obs: p.get_action(obs, deterministic=True)
                )(policy)

            else:
                continue

            policies[subtask_name][targ_name] = act_fn
            print(f"  ✓ {subtask_name}/{targ_name}")

    return policies

def act(obs, policies, uenv: SequentialTaskEnv):
    with torch.no_grad():
        with torch.device(DEVICE):
            action = torch.zeros(NUM_ENVS, *uenv.single_action_space.shape)
            obs_t  = to_tensor(obs, device=DEVICE, dtype="float")

            ptr = uenv.subtask_pointer.clone()
            subtask_type = uenv.task_ids[torch.clip(ptr, max=len(uenv.task_plan) - 1)]

            pick_idx = subtask_type == SUBTASK_TYPE_IDS["pick"]
            place_idx = subtask_type == SUBTASK_TYPE_IDS["place"]
            navigate_idx = subtask_type == SUBTASK_TYPE_IDS["navigate"]
            open_idx = subtask_type == SUBTASK_TYPE_IDS["open"]
            close_idx = subtask_type == SUBTASK_TYPE_IDS["close"]

            sapien_obj_names = [None] * NUM_ENVS
            for env_num, subtask_num in enumerate(torch.clip(ptr, max=len(uenv.task_plan) - 1)):
                subtask = uenv.task_plan[subtask_num]
                if isinstance(subtask, (PickSubtask, PlaceSubtask)):
                    sapien_obj_names[env_num] = uenv.subtask_objs[subtask_num]._objs[env_num].name
                elif isinstance(subtask, (OpenSubtask, CloseSubtask)):
                    sapien_obj_names[env_num] = uenv.subtask_articulations[subtask_num]._objs[env_num].name

            targ_names = []
            for sapien_on in sapien_obj_names:
                if sapien_on is None:
                    targ_names.append(None)
                else:
                    matched = False
                    for tn in ALL_TARG_NAMES:
                        if tn in sapien_on:
                            targ_names.append(tn)
                            matched = True
                            break
                    if not matched:
                        targ_names.append("all")

            tn_env_idxs = {}
            for env_num, tn in enumerate(targ_names):
                if tn not in tn_env_idxs:
                    tn_env_idxs[tn] = []
                tn_env_idxs[tn].append(env_num)
            for k, v in tn_env_idxs.items():
                b = torch.zeros(NUM_ENVS, dtype=torch.bool)
                b[v] = True
                tn_env_idxs[k] = b

            def set_action(subtask_name, subtask_env_idx):
                if subtask_name in ["open", "close"] or subtask_name != "navigate":
                    for tn, targ_env_idx in tn_env_idxs.items():
                        stei = subtask_env_idx & targ_env_idx
                        if torch.any(stei) and tn in policies.get(subtask_name, {}):
                            action[stei] = policies[subtask_name][tn](recursive_slice(obs_t, stei))
                            return
                if "all" in policies.get(subtask_name, {}):
                    action[subtask_env_idx] = policies[subtask_name]["all"](
                        recursive_slice(obs_t, subtask_env_idx)
                    )

            if torch.any(pick_idx):
                set_action("pick", pick_idx)
            if torch.any(place_idx):
                set_action("place", place_idx)
            if torch.any(navigate_idx):
                set_action("navigate", navigate_idx)
            if torch.any(open_idx):
                set_action("open", open_idx)
            if torch.any(close_idx):
                set_action("close", close_idx)

            return action

# -----------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------
def main():
    # Get command from argument or prompt
    if len(sys.argv) > 1:
        user_command = " ".join(sys.argv[1:])
    else:
        print("=" * 60)
        print("🤖 Single Command Execution")
        print("=" * 60)
        user_command = input("Enter command: ").strip()
        if not user_command:
            print("No command provided!")
            return

    print(f"\n{'='*70}")
    print(f"EXECUTING: {user_command}")
    print(f"{'='*70}")

    # Parse command
    print("\n[1] Parsing command...")
    parser = QwenCommandParser()
    enricher = SubtaskEnricher()
    
    simplified_subtasks = parser.parse_command(user_command)
    if not simplified_subtasks:
        print("❌ Could not parse command!")
        return
    
    subtasks = enricher.enrich_subtasks(simplified_subtasks)
    print(f"  Generated {len(subtasks)} subtasks:")
    for i, st in enumerate(subtasks, 1):
        print(f"    {i}. {st.get('type')} (obj: {st.get('obj_id', 'N/A')})")

    # Create task file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_file = OUTPUT_DIR / f"task_{timestamp}.json"
    task_plan = {
        "plans": [{
            "build_config_name": BUILD_CONFIG,
            "init_config_name": INIT_CONFIG,
            "subtasks": subtasks,
        }],
        "dataset": DATASET,
    }
    with open(task_file, "w") as f:
        json.dump(task_plan, f, indent=2)

    # Setup environment
    print("\n[2] Loading environment and policies...")
    safe_name = user_command[:30].replace(" ", "_")
    video_path = OUTPUT_DIR / f"{safe_name}_{timestamp}"
    
    env = create_env(task_file, video_path)
    obs, _ = env.reset(seed=SEED, options=dict(reconfigure=True))
    
    policies = load_policies(env, obs)

    # Execute
    print(f"\n[3] Executing ({MAX_STEPS} max steps)...")
    uenv: SequentialTaskEnv = env.unwrapped
    
    success = False
    for step in range(MAX_STEPS):
        action = act(obs, policies, uenv)
        obs, _, _, _, _ = env.step(action)
        
        current_ptr = uenv.subtask_pointer[0].item()
        
        if current_ptr >= len(subtasks):
            print(f"  ✅ Completed at step {step + 1}!")
            success = True
            break
        
        if (step + 1) % 200 == 0:
            print(f"  Step {step + 1} | Subtask {current_ptr + 1}/{len(subtasks)}")

    env.close()

    # Results
    print(f"\n{'='*70}")
    if success:
        print("✅ TASK COMPLETED SUCCESSFULLY")
    else:
        failed_idx = uenv.subtask_pointer[0].item()
        print(f"❌ TASK FAILED at subtask {failed_idx + 1}/{len(subtasks)}")
        if failed_idx < len(subtasks):
            failed_st = subtasks[failed_idx]
            print(f"   Type: {failed_st.get('type')}")
            print(f"   Object: {failed_st.get('obj_id', 'N/A')}")
    
    # Find video
    videos = sorted(video_path.parent.glob(f"{video_path.name}*.mp4"))
    if videos:
        print(f"\n📹 Video: {videos[0]}")
    else:
        print("\n⚠️  No video generated")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
