"""
Interactive Robot Live Session
- User inputs natural language commands one at a time
- Robot executes in a live simulation, preserving state between commands
- One continuous video is saved at the end
Usage:
    cd ~/ManiSkill-HAB
    python test_rafa/live_session/interactive_session.py
"""
import os
import sys
import json
import torch
import subprocess
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------
# Path setup — must happen before any mshab imports
# -----------------------------------------------------------------------
MSHAB_ROOT            = Path("/home/fri/ManiSkill-HAB")
LIVE_SESSION_DIR      = MSHAB_ROOT / "test_rafa/live_session"
INTERACTIVE_ROBOT_DIR = MSHAB_ROOT / "test_rafa/interactive_robot"

os.chdir(str(MSHAB_ROOT))
sys.path.insert(0, str(MSHAB_ROOT))
sys.path.insert(0, str(LIVE_SESSION_DIR))
sys.path.insert(0, str(INTERACTIVE_ROBOT_DIR))  # for llm_command_parser + scene_config

os.environ["SAPIEN_NO_DISPLAY"] = "1"

# -----------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------
import gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mshab.envs.make import EnvConfig, make_env
from mshab.envs import SequentialTaskEnv
from mshab.envs.planner import (
    PickSubtask, PlaceSubtask, NavigateSubtask, OpenSubtask, CloseSubtask
)
from mshab.agents.sac.agent import Agent as SACAgent
from gymnasium import spaces
from mshab.agents.sac.agent import Agent as SACAgent
from mshab.agents.ppo.agent import Agent as PPOAgent
from mshab.utils.array import to_tensor, recursive_slice
from mshab.utils.config import parse_cfg

from live_command_parser import LiveCommandParser

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

TASK_DIR   = LIVE_SESSION_DIR / "generated_tasks"
OUTPUT_DIR = LIVE_SESSION_DIR / "results"
TASK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_NAME = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SESSION_DIR  = OUTPUT_DIR / SESSION_NAME
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Subtask type IDs
SUBTASK_TYPE_IDS = {"pick": 0, "place": 1, "navigate": 2, "open": 3, "close": 4}

# Per-object policy targets for set_table
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
# Build task JSON
# -----------------------------------------------------------------------
def build_task_json(all_subtasks: list, task_file: Path):
    task_plan = {
        "plans": [{
            "build_config_name": BUILD_CONFIG,
            "init_config_name": INIT_CONFIG,
            "subtasks": all_subtasks,
        }],
        "dataset": DATASET,
    }
    with open(task_file, "w") as f:
        json.dump(task_plan, f, indent=2)

# -----------------------------------------------------------------------
# Create environment
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

# -----------------------------------------------------------------------
# Load all PPO policies for set_table
# -----------------------------------------------------------------------
def load_policies(env, dummy_obs):
    uenv: SequentialTaskEnv = env.unwrapped
    obs_space = uenv.single_observation_space
    act_space = uenv.single_action_space

    # Build pixel obs space with flattened frame stack (matches evaluate.py)
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
                print(f"  ⚠️  Missing: {subtask_name}/{targ_name}")
                continue

            # Read algo type from checkpoint's own config
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
                print(f"  ⚠️  Unknown algo '{algo_name}' for {subtask_name}/{targ_name}, skipping")
                continue

            policies[subtask_name][targ_name] = act_fn
            print(f"  ✓ {subtask_name}/{targ_name} ({algo_name})")

    return policies

# -----------------------------------------------------------------------
# Act function — mirrors evaluate.py logic exactly
# -----------------------------------------------------------------------
def act(obs, policies, uenv: SequentialTaskEnv):
    with torch.no_grad():
        with torch.device(DEVICE):
            action = torch.zeros(NUM_ENVS, *uenv.single_action_space.shape)
            obs_t  = to_tensor(obs, device=DEVICE, dtype="float")

            ptr         = uenv.subtask_pointer.clone()
            subtask_type = uenv.task_ids[
                torch.clip(ptr, max=len(uenv.task_plan) - 1)
            ]

            pick_idx     = subtask_type == SUBTASK_TYPE_IDS["pick"]
            place_idx    = subtask_type == SUBTASK_TYPE_IDS["place"]
            navigate_idx = subtask_type == SUBTASK_TYPE_IDS["navigate"]
            open_idx     = subtask_type == SUBTASK_TYPE_IDS["open"]
            close_idx    = subtask_type == SUBTASK_TYPE_IDS["close"]

            # Determine target names per env
            sapien_obj_names = [None] * NUM_ENVS
            for env_num, subtask_num in enumerate(
                torch.clip(ptr, max=len(uenv.task_plan) - 1)
            ):
                subtask = uenv.task_plan[subtask_num]
                if isinstance(subtask, (PickSubtask, PlaceSubtask)):
                    sapien_obj_names[env_num] = (
                        uenv.subtask_objs[subtask_num]._objs[env_num].name
                    )
                elif isinstance(subtask, (OpenSubtask, CloseSubtask)):
                    sapien_obj_names[env_num] = (
                        uenv.subtask_articulations[subtask_num]._objs[env_num].name
                    )

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

            # Build per-target bool index maps
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
                            action[stei] = policies[subtask_name][tn](
                                recursive_slice(obs_t, stei)
                            )
                            return
                # fallback to "all"
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
# Run simulation until new subtasks complete or max steps reached
# -----------------------------------------------------------------------
def run_until_done(env, policies, start_subtask_idx: int, total_subtasks: int):
    uenv: SequentialTaskEnv = env.unwrapped

    obs, _ = env.reset(seed=SEED, options=dict(reconfigure=True))

    # Advance pointer past already-completed subtasks
    if start_subtask_idx > 0:
        print(f"  ⏩ Resuming from subtask {start_subtask_idx + 1}/{total_subtasks}")
        uenv.subtask_pointer = torch.full(
            (NUM_ENVS,), start_subtask_idx, dtype=torch.long,
            device=uenv.subtask_pointer.device,
        )

    success = False

    print(f"  🤖 Running subtasks {start_subtask_idx + 1} → {total_subtasks}...")

    for step in range(MAX_STEPS):
        action = act(obs, policies, uenv)
        obs, _, _, _, _ = env.step(action)

        current_ptr = uenv.subtask_pointer[0].item()

        if current_ptr >= total_subtasks:
            print(f"  ✅ Completed at step {step + 1}!")
            success = True
            break

        if (step + 1) % 200 == 0:
            print(f"  📊 Step {step + 1}/{MAX_STEPS} | Subtask {current_ptr + 1}/{total_subtasks}")

    if not success:
        print(f"  ⚠️  Max steps reached without full completion")

    return success, uenv.subtask_pointer[0].item()

# -----------------------------------------------------------------------
# Combine video clips into one final video
# -----------------------------------------------------------------------
def combine_videos(video_files: list, output_path: Path):
    if not video_files:
        print("No videos to combine.")
        return
    if len(video_files) == 1:
        import shutil
        shutil.copy(video_files[0], str(output_path))
        print(f"  📹 Video: {output_path}")
        return

    concat_file = SESSION_DIR / "concat_list.txt"
    with open(concat_file, "w") as f:
        for vf in video_files:
            f.write(f"file '{vf}'\n")

    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_file), "-c", "copy", str(output_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  📹 Final video: {output_path}")
    else:
        print(f"  ⚠️  ffmpeg error: {result.stderr}")

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("🤖 ManiSkill-HAB Live Interactive Session")
    print("=" * 60)
    print("  Enter a natural language command to execute")
    print("  'done'  → finish and save final video")
    print("  'quit'  → exit without saving")
    print("=" * 60)

    # ── Bootstrap: create a minimal env to get obs/act spaces for policy loading
    print("\n📦 Loading LLM parser...")
    parser = LiveCommandParser()

    print("\n📦 Building bootstrap env to load policies...")
    bootstrap_subtasks = [
        {"type": "navigate", "target": "024_bowl-0"},
        {"type": "pick",     "obj_id": "024_bowl-0"},
    ]
    bootstrap_task_file = TASK_DIR / "bootstrap_task.json"
    build_task_json(bootstrap_subtasks, bootstrap_task_file)

    bootstrap_env = create_env(bootstrap_task_file, SESSION_DIR / "bootstrap")
    uenv_boot: SequentialTaskEnv = bootstrap_env.unwrapped
    dummy_obs, _ = bootstrap_env.reset(seed=SEED, options=dict(reconfigure=True))
    dummy_obs_t  = to_tensor(dummy_obs, device=DEVICE, dtype="float")
    act_space    = uenv_boot.single_action_space

    print("\n📦 Loading policies...")
    policies = load_policies(bootstrap_env, dummy_obs)
    #policies = load_policies(dummy_obs_t, act_space)
    bootstrap_env.close()
    print("✅ Ready!\n")

    # ── Session state
    all_subtasks  = []
    completed_ptr = 0
    video_files   = []
    command_count = 0

    # ── Interactive loop
    while True:
        print(f"\n{'─' * 50}")
        user_input = input("🗣️  Command: ").strip()

        if user_input.lower() == "quit":
            print("Exiting without saving.")
            break

        if user_input.lower() == "done":
            print("\nSaving session...")
            combine_videos(video_files, SESSION_DIR / "final_session.mp4")
            print(f"\n✅ Done! {command_count} command(s) executed.")
            print(f"   Session folder: {SESSION_DIR}")
            break

        if not user_input:
            continue

        command_count += 1
        print(f"\n[Command {command_count}] '{user_input}'")

        # 1. Parse
        new_subtasks = parser.parse_command(user_input)
        if not new_subtasks:
            print("  ❌ Could not parse. Try rephrasing.")
            command_count -= 1
            continue
        print(f"  📋 {len(new_subtasks)} subtask(s) generated")

        # 2. Accumulate
        all_subtasks  += new_subtasks
        total_subtasks = len(all_subtasks)

        # 3. Write task JSON with ALL subtasks
        task_file = TASK_DIR / f"set_table_task_{SESSION_NAME}_cmd{command_count:02d}.json"
        build_task_json(all_subtasks, task_file)

        # 4. Create env
        safe_name  = user_input[:25].replace(" ", "_")
        video_path = SESSION_DIR / f"cmd_{command_count:02d}_{safe_name}"
        env = create_env(task_file, video_path)

        # 5. Run
        success, new_ptr = run_until_done(
            env, policies,
            start_subtask_idx=completed_ptr,
            total_subtasks=total_subtasks,
        )

        # 6. Collect clip
        clips = sorted(video_path.parent.glob(f"{video_path.name}*.mp4"))
        if clips:
            video_files.append(str(clips[-1]))
            print(f"  🎬 Clip: {clips[-1].name}")

        # 7. Update state
        completed_ptr = new_ptr
        env.close()

        # 8. Sync parser object locations
        for subtask in new_subtasks:
            stype = subtask.get("type")
            obj   = subtask.get("obj_id", "")
            if stype == "pick":
                parser.update_object_location(obj, "held")
            elif stype == "place":
                parser.update_object_location(obj, subtask.get("target", "counter"))

        status = "✅ Success" if success else "⚠️  Partial"
        print(f"  {status} | {completed_ptr}/{total_subtasks} subtasks done")


if __name__ == "__main__":
    main()
