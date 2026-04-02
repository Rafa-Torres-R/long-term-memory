#!/usr/bin/bash

# evaluate_custom_task.sh - Run custom apple-to-drawer task with trained policies

SEED=0
TASK=set_table
policy_type=rl ## Change this to your trained policy type (e.g., rl, rl_per_obj, bc, etc.)  
#policy_key=rl_per_obj 
continuous_task=True

# VIDEO OUTPUT - CUSTOMIZE THIS!
WORKSPACE="/home/fri/ManiSkill-HAB/test_rafa/evaluation_results/appletodrawer"  # Change to wherever you want
GROUP=eval_custom_task
EXP_NAME="apple_in_drawer_$(date +%Y%m%d_%H%M%S)"  # Adds timestamp and name of what you re doing

record_video=True
info_on_video=False #true will show info on video, false will not show info on video, can change to true if you want to see the info on the video but it may be distracting when watching the video so set to false for final evaluation videos
save_trajectory=False
max_trajectories=1

if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi

WANDB=False
TENSORBOARD=True
PROJECT_NAME="MS-HAB-Custom-AppleDrawer"

# Environment settings
ENV_ID="SequentialTask-v0"
NUM_ENVS=9
MAX_EPISODE_STEPS=200 #used to use 7000 but changed to 1500 to save time for testing, can change back to 7000 for final evaluation 
#(20 steps is 1 sec aprox)
FRAME_STACK=3
STACK=null

# Custom task plan
CUSTOM_TASK_PLAN_FP="$HOME/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/sequential/train/custom_apple_drawer.json"

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
    eval_env.task_plan_fp="$CUSTOM_TASK_PLAN_FP" \
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
    logger.wandb="$WANDB" \
    logger.tensorboard="$TENSORBOARD" \
    logger.project_name="$PROJECT_NAME" \
    logger.workspace="$WORKSPACE" \
    logger.clear_out="False" \
    logger.wandb_cfg.group="$GROUP" \
    logger.exp_name="$EXP_NAME" \
    "${extra_args[@]}"

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "Videos saved to: $WORKSPACE/$EXP_NAME/"
echo "============================================"
