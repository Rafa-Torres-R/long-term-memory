cat > ~/ManiSkill-HAB/test_rafa/interactive_robot/llm_command_parser.py << 'EOF'
"""
LLM Command Parser with Hard-coded Logic for Complex Cases
"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from scene_config import OBJECTS, ARTICULATIONS, FRIENDLY_NAME_TO_OBJ_ID

class QwenCommandParser:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Loading Qwen model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print(f"Model loaded on: {self.model.device}")
        
        self.object_locations = {
            obj_id: obj_info["initial_location"] 
            for obj_id, obj_info in OBJECTS.items()
        }
        
        self.object_positions = {
            obj_id: obj_info["initial_position"] 
            for obj_id, obj_info in OBJECTS.items()
        }
    
    def parse_command(self, user_command):
        swap_result = self._detect_and_handle_swap(user_command)
        if swap_result:
            print("🔧 Detected swap command - using hard-coded staging logic")
            return self._translate_friendly_names(swap_result)
        
        drawer_pick_result = self._detect_and_handle_pick_from_drawer(user_command)
        if drawer_pick_result:
            print("🔧 Detected pick from drawer - using hard-coded open sequence")
            return self._translate_friendly_names(drawer_pick_result)
        
        scene_context = self._build_scene_context()
        system_prompt = self._build_system_prompt(scene_context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Command: {user_command}"}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
        
        input_length = inputs.input_ids.shape[-1]
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        subtasks = self._extract_json(response)
        return self._translate_friendly_names(subtasks)
    
    def _detect_and_handle_pick_from_drawer(self, command):
        command_lower = command.lower()
        
        if not any(word in command_lower for word in ["pick", "get", "grab", "take"]):
            return None
        
        # ⭐ FIX: Check for both "bowl_3" and "bowl 3"
        for friendly_name, obj_id in FRIENDLY_NAME_TO_OBJ_ID.items():
            # Create variants: "bowl_3" → ["bowl_3", "bowl 3"]
            name_variants = [
                friendly_name,
                friendly_name.replace("_", " ")  # Replace underscore with space
            ]
            
            # Check if ANY variant is in the command
            if any(variant in command_lower for variant in name_variants):
                # Check if this object is in the drawer
                if self.object_locations[obj_id] == "drawer":
                    print(f"  ℹ️  {friendly_name} is inside drawer - adding open sequence")
                    
                    return [
                        {"type": "navigate", "target": "drawer"},
                        {"type": "open", "articulation": "drawer", "obj_id": friendly_name},
                        {"type": "navigate", "target": friendly_name},
                        {"type": "pick", "obj_id": friendly_name, "from": "drawer"}
                    ]
        
        return None
    
    def _translate_friendly_names(self, subtasks):
        for subtask in subtasks:
            if "obj_id" in subtask:
                friendly = subtask["obj_id"]
                if friendly in FRIENDLY_NAME_TO_OBJ_ID:
                    subtask["obj_id"] = FRIENDLY_NAME_TO_OBJ_ID[friendly]
        return subtasks
    
    def _detect_and_handle_swap(self, command):
        command_lower = command.lower()
        if not any(word in command_lower for word in ["swap", "switch", "exchange"]):
            return None
        
        obj_ids = []
        for friendly_name in FRIENDLY_NAME_TO_OBJ_ID.keys():
            # ⭐ Check both formats
            name_variants = [friendly_name, friendly_name.replace("_", " ")]
            if any(variant in command for variant in name_variants):
                obj_ids.append(friendly_name)
        
        if len(obj_ids) != 2:
            return None
        
        obj_a, obj_b = obj_ids[0], obj_ids[1]
        actual_a = FRIENDLY_NAME_TO_OBJ_ID[obj_a]
        actual_b = FRIENDLY_NAME_TO_OBJ_ID[obj_b]
        pos_a = self.object_positions[actual_a]
        pos_b = self.object_positions[actual_b]
        
        if not pos_a or not pos_b:
            return None
        
        staging_pos = [0.0, -2.0, 1.05]
        
        return [
            {"type": "navigate", "target": obj_a},
            {"type": "pick", "obj_id": obj_a},
            {"type": "navigate", "target": "staging"},
            {"type": "place", "obj_id": obj_a, "target": "counter", "goal_pos": staging_pos},
            {"type": "navigate", "target": obj_b},
            {"type": "pick", "obj_id": obj_b},
            {"type": "navigate", "target": f"{obj_a}_original"},
            {"type": "place", "obj_id": obj_b, "target": "counter", "goal_pos": pos_a},
            {"type": "navigate", "target": "staging"},
            {"type": "pick", "obj_id": obj_a},
            {"type": "navigate", "target": f"{obj_b}_original"},
            {"type": "place", "obj_id": obj_a, "target": "counter", "goal_pos": pos_b},
        ]
    
    def _build_scene_context(self):
        context = {
            "objects": {},
            "articulations": {
                name: {"type": config["articulation_type"]}
                for name, config in ARTICULATIONS.items()
            }
        }
        
        for obj_id, obj_info in OBJECTS.items():
            context["objects"][obj_info["friendly_name"]] = {
                "type": obj_info["type"],
                "current_location": self.object_locations[obj_id],
                "position": self.object_positions[obj_id],
                "accessible": self.object_locations[obj_id] != "drawer"
            }
        
        return context
    
    def _build_system_prompt(self, scene_context):
        prompt = f"""You are a robot task planner. Convert user commands into subtask sequences.

Available Objects: apple, bowl_0, bowl_1, bowl_3

Available actions:
- navigate, pick, place, open, close

RULES:
1. ALWAYS navigate before pick/place/open/close
2. Can only hold ONE object
3. Cannot open/close while holding object
4. When placing on counter, specify exact goal_pos

Example - Pick:
[
  {{"type": "navigate", "target": "bowl_0"}},
  {{"type": "pick", "obj_id": "bowl_0"}}
]

Example - Put in drawer:
[
  {{"type": "navigate", "target": "apple"}},
  {{"type": "pick", "obj_id": "apple"}},
  {{"type": "navigate", "target": "drawer"}},
  {{"type": "open", "articulation": "drawer", "obj_id": "apple"}},
  {{"type": "navigate", "target": "drawer_inside"}},
  {{"type": "place", "obj_id": "apple", "target": "drawer"}},
  {{"type": "navigate", "target": "away_from_drawer"}},
  {{"type": "close", "articulation": "drawer"}}
]

Respond ONLY with JSON array."""
        return prompt
    
    def _extract_json(self, response):
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
        return []
    
    def update_object_location(self, obj_id, new_location, new_position=None):
        if obj_id in self.object_locations:
            self.object_locations[obj_id] = new_location
        if new_position and obj_id in self.object_positions:
            self.object_positions[obj_id] = new_position
EOF
