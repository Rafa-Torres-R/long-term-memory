
# class InteractiveRobotExecutor:
#     def execute_command(user_command):
#         # 1. Parse with LLM
#         # 2. Enrich subtasks
#         # 3. Create task plan JSON
#         # 4. Run simulation

"""
Interactive Robot Executor V2 - Uses existing evaluation infrastructure
"""
import sys
sys.path.insert(0, '/home/fri/ManiSkill-HAB')

import json
import subprocess
from pathlib import Path
from datetime import datetime
from llm_command_parser import QwenCommandParser
from subtask_enricher import SubtaskEnricher

class InteractiveRobotExecutor:
    def __init__(self):
        print("="*70)
        print("INTERACTIVE ROBOT EXECUTOR V2")
        print("="*70)
        
        print("\n[1] Loading LLM parser...")
        self.parser = QwenCommandParser()
        self.enricher = SubtaskEnricher()
        
        # FIXED: Put task plans in a path that contains "set_table"
        self.task_plans_dir = Path("/home/fri/ManiSkill-HAB/test_rafa/interactive_robot/task_plans/set_table")
        self.task_plans_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_script = Path("/home/fri/ManiSkill-HAB/test_rafa/interactive_robot/run_command.sh")
        
        print("\n✅ Executor initialized!")
    
    def _create_task_plan_file(self, enriched_subtasks, command_name):
        """Create task plan JSON file"""
        task_plan = {
            "plans": [
                {
                    "build_config_name": "v3_sc1_staging_04.scene_instance.json",
                    "init_config_name": "train/set_table/episode_0.json",
                    "subtasks": enriched_subtasks
                }
            ],
            "dataset": "ReplicaCADRearrangeDataset"
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{command_name}_{timestamp}.json"
        filepath = self.task_plans_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(task_plan, f, indent=2)
        
        print(f"    ✅ Created task plan: {filepath}")
        return filepath
    
    def execute_command(self, user_command, run_simulation=True):
        """Execute a natural language command"""
        print(f"\n{'='*70}")
        print(f"EXECUTING: {user_command}")
        print(f"{'='*70}")
        
        # Step 1: Parse command with LLM
        print("\n[Step 1] LLM parsing command...")
        simplified_subtasks = self.parser.parse_command(user_command)
        print(f"  Generated {len(simplified_subtasks)} subtasks")
        
        # DEBUG: Show what the LLM actually generated
        print("\n  📋 LLM Output (Simplified):")
        for i, subtask in enumerate(simplified_subtasks, 1):
            print(f"    {i}. {subtask}")
        
        # Step 2: Enrich with full details
        print("\n[Step 2] Enriching subtasks...")
        enriched_subtasks = self.enricher.enrich_subtasks(simplified_subtasks)
        print(f"  ✅ Enriched with full ManiSkill format")
        
        # DEBUG: Show enriched subtasks
        print("\n  📋 Enriched Subtasks:")
        for i, subtask in enumerate(enriched_subtasks, 1):
            task_type = subtask.get('type', 'unknown')
            obj_id = subtask.get('obj_id', 'N/A')
            print(f"    {i}. {task_type} (obj: {obj_id})")
        
        # Step 3: Create task plan file
        print("\n[Step 3] Creating task plan...")
        command_name = user_command.replace(" ", "_")[:30]  # Sanitize filename
        task_plan_path = self._create_task_plan_file(enriched_subtasks, command_name)
        
        if not run_simulation:
            print("\n⚠️  Simulation disabled - task plan created but not executed")
            return enriched_subtasks
        
        # Step 4: Execute using ManiSkill evaluation infrastructure
        print("\n[Step 4] Running simulation...")
        print("  This may take a few minutes...")
        
        try:
            result = subprocess.run(
                [str(self.run_script), str(task_plan_path), command_name],
                cwd=Path.home() / "ManiSkill-HAB",
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            print(result.stdout)
            
            if result.returncode == 0:
                print("\n✅ Simulation completed successfully!")
                # Find and show the video location
                workspace = Path("/home/fri/ManiSkill-HAB/test_rafa/evaluation_results/interactive_commands")
                latest_runs = sorted(workspace.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if latest_runs:
                    print(f"\n📹 Check for videos in: {latest_runs[0]}/")
            else:
                print(f"\n❌ Simulation failed with code {result.returncode}")
                print(result.stderr)
        
        except subprocess.TimeoutExpired:
            print("\n⚠️  Simulation timed out after 10 minutes")
        except Exception as e:
            print(f"\n❌ Error running simulation: {e}")
        
        return enriched_subtasks

def main():
    """Test the executor"""
    executor = InteractiveRobotExecutor()
    
    print("\n" + "="*70)
    print("INTERACTIVE ROBOT CONTROL V2")
    print("="*70)
    print("This version uses the existing ManiSkill evaluation infrastructure")
    print("="*70 + "\n")
    
    # Test commands - you can modify these to test different scenarios
    test_command = "pick up bowl_3"
    print(f"Testing with: '{test_command}'\n")
    
    executor.execute_command(test_command, run_simulation=True)  # Set to False to just see subtasks

if __name__ == "__main__":
    main()
