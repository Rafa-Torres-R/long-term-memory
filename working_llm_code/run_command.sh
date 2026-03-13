#!/usr/bin/bash
# Run a single interactive command using ManiSkill-HAB evaluation infrastructure

TASK_PLAN_FILE="$1"  # Path to task plan JSON
OUTPUT_NAME="$2"     # Name for this run

if [[ -z "$TASK_PLAN_FILE" ]]; then
    echo "Usage: ./run_command.sh <task_plan_file> <output_name>"
    exit 1
fi

# Configuration
SEED=0
TASK=set_table
policy_type=rl
continuous_task=True

# Output settings
WORKSPACE="/home/fri/ManiSkill-HAB/test_rafa/evaluation_results/interactive_commands"
GROUP=interactive_commands
EXP_NAME="${OUTPUT_NAME}_$(date +%Y%m%d_%H%M%S)"

# Video settings
record_video=True
info_on_video=False
save_trajectory=False
max_trajectories=1

# Environment settings
ENV_ID="SequentialTask-v0"
NUM_ENVS=9  # Single environment for interactive use
MAX_EPISODE_STEPS=300 # Adjust as needed for your command
FRAME_STACK=3
STACK=null

# Scene builder
SCENE_BUILDER="ReplicaCADSetTableTrain"

# Extra args
extra_args=(
    "eval_env.env_kwargs.task_cfgs.navigate.ignore_arm_checkers=True"
    "eval_env.env_kwargs.scene_builder_cls=$SCENE_BUILDER"
)

# Run evaluation
SAPIEN_NO_DISPLAY=1 python -m mshab.evaluate configs/evaluate.yml \
    seed=$SEED \
    task=$TASK \
    save_trajectory=$save_trajectory \
    max_trajectories=$max_trajectories \
    policy_type=$policy_type \
    eval_env.env_id="$ENV_ID" \
    eval_env.task_plan_fp="$TASK_PLAN_FILE" \
    eval_env.make_env="True" \
    eval_env.num_envs=$NUM_ENVS \
    eval_env.frame_stack=$FRAME_STACK \
    eval_env.stack=$STACK \
    eval_env.max_episode_steps=$MAX_EPISODE_STEPS \
    eval_env.continuous_task=$continuous_task \
    eval_env.record_video="$record_video" \
    eval_env.info_on_video="$info_on_video" \
    eval_env.save_video_freq=1 \
    logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
    logger.wandb="False" \
    logger.tensorboard="True" \
    logger.project_name="Interactive-Commands" \
    logger.workspace="$WORKSPACE" \
    logger.clear_out="False" \
    logger.wandb_cfg.group="$GROUP" \
    logger.exp_name="$EXP_NAME" \
    "${extra_args[@]}"

echo ""
echo "============================================"
echo "Command execution complete!"
echo "Videos saved to: $WORKSPACE/$EXP_NAME/"
echo "============================================"
