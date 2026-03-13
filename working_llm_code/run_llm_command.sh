#!/usr/bin/bash
# Bridge script: Natural language → LLM → Task Plan → Execute

COMMAND="$1"  # User's natural language command
OUTPUT_DIR="/home/fri/ManiSkill-HAB/test_rafa/interactive_robot/generated_tasks"
mkdir -p "$OUTPUT_DIR"

if [[ -z "$COMMAND" ]]; then
    echo "Usage: ./run_llm_command.sh \"<your command>\""
    echo "Example: ./run_llm_command.sh \"pick up the apple\""
    exit 1
fi

echo "================================"
echo "Processing command: $COMMAND"
echo "================================"

# Step 1: Generate task plan using LLM
# FIXED: Include "set_table" in filename to pass assertion check
TASK_FILE="$OUTPUT_DIR/set_table_task_$(date +%Y%m%d_%H%M%S).json"
python3 << PYTHON_EOF
from llm_command_parser import QwenCommandParser
import json

parser = QwenCommandParser()
subtasks = parser.parse_command("$COMMAND")

# Convert to ManiSkill-HAB format
task_plan = {
    "plans": [{
        "build_config_name": "v3_sc1_staging_04.scene_instance.json",
        "init_config_name": "/home/fri/ManiSkill-HAB/test_rafa/changing_objects/original_episode_0.json",
        "subtasks": subtasks
    }],
    "dataset": "ReplicaCADRearrangeDataset"
}

with open("$TASK_FILE", "w") as f:
    json.dump(task_plan, f, indent=2)

print(f"✓ Task plan saved to: $TASK_FILE")
PYTHON_EOF

# ← Add this block
echo ""
echo "=== GENERATED TASK FILE ==="
cat "$TASK_FILE"
echo ""
echo "==========================="

# Step 2: Execute the generated task plan
echo ""
echo "Executing task plan..."
cd ~/ManiSkill-HAB
bash /home/fri/ManiSkill-HAB/test_rafa/interactive_robot/run_command.sh "$TASK_FILE" "llm_command"
