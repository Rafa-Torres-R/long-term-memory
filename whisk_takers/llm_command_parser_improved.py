"""
LLM Command Parser with Hard-coded Logic for Complex Cases
"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from scene_config import OBJECTS, ARTICULATIONS, FRIENDLY_NAME_TO_OBJ_ID, COUNTER_X, COUNTER_Y, COUNTER_Z , DRAWER_CENTER_X, DRAWER_CENTER_Y, DRAWER_CENTER_Z

class QwenCommandParser:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
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
        print(f"  🔍 object_locations: {self.object_locations}")
        
        swap_result = self._detect_and_handle_swap(user_command)
        if swap_result:
            print("🔧 Using: swap handler")
            return self._translate_friendly_names(swap_result)
        
        # Check place-only-in-container BEFORE pick-and-place
        place_only_in_container_result = self._detect_and_handle_place_only_in_container(user_command)
        if place_only_in_container_result:
            print("🔧 Using: place-only in container handler")
            return self._translate_friendly_names(place_only_in_container_result)
        
        place_in_container_result = self._detect_and_handle_pick_and_place_in_container(user_command)
        if place_in_container_result:
            print("🔧 Using: pick and place in container handler")
            return self._translate_friendly_names(place_in_container_result)
        
        pick_and_place_result = self._detect_and_handle_pick_and_place(user_command)
        if pick_and_place_result:
            print("🔧 Using: pick and place handler")
            return self._translate_friendly_names(pick_and_place_result)
        
        container_pick_result = self._detect_and_handle_pick_from_container(user_command)
        if container_pick_result:
            print("🔧 Using: container pick handler")
            return self._translate_friendly_names(container_pick_result)
        
        print("🔧 Using: LLM fallback")
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
    
    # -----------------------------------------------------------------------
    # Handler: place in container (object already held)
    # e.g. "place bowl 3 in drawer" (assumes robot is holding bowl)
    # -----------------------------------------------------------------------
    def _detect_and_handle_place_only_in_container(self, command):
        command_lower = command.lower().replace("_", " ")
        
        # Must have "place" but NOT "pick"
        has_place = any(word in command_lower for word in ["put", "place", "set"])
        has_pick = any(word in command_lower for word in ["pick", "get", "grab", "take"])
        has_container = any(word in command_lower for word in ["drawer", "fridge"])
        
        if not (has_place and has_container and not has_pick):
            return None
        
        articulation = "drawer" if "drawer" in command_lower else "fridge"
        
        for friendly_name, obj_id in FRIENDLY_NAME_TO_OBJ_ID.items():
            normalized_friendly = friendly_name.lower().replace("_", " ")
            if normalized_friendly in command_lower:
                print(f"  ℹ️  Placing {friendly_name} into {articulation} (assuming already held)")
                
                return [
                    # Navigate to container
                    {"type": "navigate", "target": articulation},
                    # Open container
                    {
                        "type": "open",
                        "articulation_type":                    ARTICULATIONS[articulation]["articulation_type"],
                        "articulation_id":                      ARTICULATIONS[articulation]["articulation_id"],
                        "articulation_handle_link_idx":         ARTICULATIONS[articulation]["articulation_handle_link_idx"],
                        "articulation_handle_active_joint_idx": ARTICULATIONS[articulation]["articulation_handle_active_joint_idx"],
                        "articulation_relative_handle_pos":     ARTICULATIONS[articulation]["articulation_relative_handle_pos"],
                    },
                    # Navigate again (to align for placement)
                    {"type": "navigate", "target": articulation},
                    # Place inside
                    {
                        "type": "place",
                        "obj_id": obj_id,
                        "goal_pos": [DRAWER_CENTER_X, DRAWER_CENTER_Y, DRAWER_CENTER_Z],
                        "goal_rectangle_corners": [
                            [DRAWER_CENTER_X + 0.001, DRAWER_CENTER_Y - 0.001, DRAWER_CENTER_Z],
                            [DRAWER_CENTER_X + 0.001, DRAWER_CENTER_Y + 0.001, DRAWER_CENTER_Z],
                            [DRAWER_CENTER_X - 0.001, DRAWER_CENTER_Y - 0.001, DRAWER_CENTER_Z],
                            [DRAWER_CENTER_X - 0.001, DRAWER_CENTER_Y + 0.001, DRAWER_CENTER_Z],
                        ],
                        "validate_goal_rectangle_corners": False,
                        "articulation_config": None,
                        "subtask_config": {"obj_goal_thresh": 0.5},
                    },
                    # Navigate to close
                    {"type": "navigate", "target": articulation},
                    # Close container (hands free now!)
                    {
                        "type": "close",
                        "articulation_type":                    ARTICULATIONS[articulation]["articulation_type"],
                        "articulation_id":                      ARTICULATIONS[articulation]["articulation_id"],
                        "articulation_handle_link_idx":         ARTICULATIONS[articulation]["articulation_handle_link_idx"],
                        "articulation_handle_active_joint_idx": ARTICULATIONS[articulation]["articulation_handle_active_joint_idx"],
                        "articulation_relative_handle_pos":     ARTICULATIONS[articulation]["articulation_relative_handle_pos"],
                    },
                ]
        
        return None
    
    # -----------------------------------------------------------------------
    # Handler: pick from container only (e.g. "pick up the apple")
    # Does NOT close container after picking
    # -----------------------------------------------------------------------
    def _detect_and_handle_pick_from_container(self, command):
        command_lower = command.lower().replace("_", " ")
        
        if not any(word in command_lower for word in ["pick", "get", "grab", "take"]):
            return None
        
        CONTAINER_ARTICULATIONS = {
            "drawer": "drawer",
            "fridge": "fridge",
        }
        
        for friendly_name, obj_id in FRIENDLY_NAME_TO_OBJ_ID.items():
            normalized_friendly = friendly_name.lower().replace("_", " ")
            if normalized_friendly in command_lower:
                location = self.object_locations[obj_id]
                
                if location in CONTAINER_ARTICULATIONS:
                    articulation = CONTAINER_ARTICULATIONS[location]
                    print(f"  ℹ️  {friendly_name} is inside {articulation} - adding open sequence")
                    
                    return [
                        {"type": "navigate", "target": articulation},
                        {
                            "type": "open",
                            "articulation_type":                    ARTICULATIONS[articulation]["articulation_type"],
                            "articulation_id":                      ARTICULATIONS[articulation]["articulation_id"],
                            "articulation_handle_link_idx":         ARTICULATIONS[articulation]["articulation_handle_link_idx"],
                            "articulation_handle_active_joint_idx": ARTICULATIONS[articulation]["articulation_handle_active_joint_idx"],
                            "articulation_relative_handle_pos":     ARTICULATIONS[articulation]["articulation_relative_handle_pos"],
                        },
                        {"type": "navigate", "target": normalized_friendly},
                        {"type": "pick", "obj_id": obj_id},
                        # NO CLOSE - drawer stays open!
                    ]
        
        return None
    
    # -----------------------------------------------------------------------
    # Handler: pick and place on counter (e.g. "pick bowl 3 and place on counter")
    # WILL close container after placing
    # -----------------------------------------------------------------------
    def _detect_and_handle_pick_and_place(self, command):
        command_lower = command.lower().replace("_", " ")
        
        has_pick = any(word in command_lower for word in ["pick", "get", "grab", "take"])
        has_place = any(word in command_lower for word in ["put", "place", "set"])
        has_counter = any(word in command_lower for word in ["counter", "table", "surface"])
        
        if not (has_pick and has_place and has_counter):
            return None
        
        for friendly_name, obj_id in FRIENDLY_NAME_TO_OBJ_ID.items():
            normalized_friendly = friendly_name.lower().replace("_", " ")
            if normalized_friendly in command_lower:
                location = self.object_locations[obj_id]
                subtasks = []
                
                # If object is in a container, open it first
                if location in ["drawer", "fridge"]:
                    articulation = location
                    subtasks += [
                        {"type": "navigate", "target": articulation},
                        {
                            "type": "open",
                            "articulation_type":                    ARTICULATIONS[articulation]["articulation_type"],
                            "articulation_id":                      ARTICULATIONS[articulation]["articulation_id"],
                            "articulation_handle_link_idx":         ARTICULATIONS[articulation]["articulation_handle_link_idx"],
                            "articulation_handle_active_joint_idx": ARTICULATIONS[articulation]["articulation_handle_active_joint_idx"],
                            "articulation_relative_handle_pos":     ARTICULATIONS[articulation]["articulation_relative_handle_pos"],
                        },
                    ]
                
                # Pick the object
                subtasks += [
                    {"type": "navigate", "target": normalized_friendly},
                    {"type": "pick", "obj_id": obj_id},
                ]
                
                # Place on counter
                subtasks += [
                    {"type": "navigate", "target": "counter"},
                    {
                        "type": "place",
                        "obj_id": obj_id,
                        "goal_pos": [COUNTER_X, COUNTER_Y, COUNTER_Z],
                        "goal_rectangle_corners": [
                            [COUNTER_X + 0.001, COUNTER_Y - 0.001, COUNTER_Z],
                            [COUNTER_X + 0.001, COUNTER_Y + 0.001, COUNTER_Z],
                            [COUNTER_X - 0.001, COUNTER_Y - 0.001, COUNTER_Z],
                            [COUNTER_X - 0.001, COUNTER_Y + 0.001, COUNTER_Z],
                        ],
                        "validate_goal_rectangle_corners": False,
                        "articulation_config": None,
                    },
                ]
                
                # Close container (hands free now!)
                if location in ["drawer", "fridge"]:
                    subtasks += [
                        {"type": "navigate", "target": articulation},
                        {
                            "type": "close",
                            "articulation_type":                    ARTICULATIONS[articulation]["articulation_type"],
                            "articulation_id":                      ARTICULATIONS[articulation]["articulation_id"],
                            "articulation_handle_link_idx":         ARTICULATIONS[articulation]["articulation_handle_link_idx"],
                            "articulation_handle_active_joint_idx": ARTICULATIONS[articulation]["articulation_handle_active_joint_idx"],
                            "articulation_relative_handle_pos":     ARTICULATIONS[articulation]["articulation_relative_handle_pos"],
                        },
                    ]
                
                return subtasks
        
        return None
    
    # -----------------------------------------------------------------------
    # Handler: pick and place in container (e.g. "pick bowl 3 and place in drawer")
    # -----------------------------------------------------------------------
    def _detect_and_handle_pick_and_place_in_container(self, command):
        command_lower = command.lower().replace("_", " ")
        
        has_pick = any(word in command_lower for word in ["pick", "get", "grab", "take"])
        has_place = any(word in command_lower for word in ["put", "place", "set"])
        has_container = any(word in command_lower for word in ["drawer", "fridge"])
        
        if not (has_pick and has_place and has_container):
            return None
        
        articulation = "drawer" if "drawer" in command_lower else "fridge"
        
        for friendly_name, obj_id in FRIENDLY_NAME_TO_OBJ_ID.items():
            normalized_friendly = friendly_name.lower().replace("_", " ")
            if normalized_friendly in command_lower:
                print(f"  ℹ️  Placing {friendly_name} into {articulation}")
                
                return [
                    # Open container first (hands free)
                    {"type": "navigate", "target": articulation},
                    {
                        "type": "open",
                        "articulation_type":                    ARTICULATIONS[articulation]["articulation_type"],
                        "articulation_id":                      ARTICULATIONS[articulation]["articulation_id"],
                        "articulation_handle_link_idx":         ARTICULATIONS[articulation]["articulation_handle_link_idx"],
                        "articulation_handle_active_joint_idx": ARTICULATIONS[articulation]["articulation_handle_active_joint_idx"],
                        "articulation_relative_handle_pos":     ARTICULATIONS[articulation]["articulation_relative_handle_pos"],
                    },
                    # Pick the object
                    {"type": "navigate", "target": normalized_friendly},
                    {"type": "pick", "obj_id": obj_id},
                    # Place inside container
                    {"type": "navigate", "target": articulation},
                    {
                        "type": "place",
                        "obj_id": obj_id,
                        "goal_pos": [DRAWER_CENTER_X, DRAWER_CENTER_Y, DRAWER_CENTER_Z],
                        "goal_rectangle_corners": [
                            [DRAWER_CENTER_X + 0.001, DRAWER_CENTER_Y - 0.001, DRAWER_CENTER_Z],
                            [DRAWER_CENTER_X + 0.001, DRAWER_CENTER_Y + 0.001, DRAWER_CENTER_Z],
                            [DRAWER_CENTER_X - 0.001, DRAWER_CENTER_Y - 0.001, DRAWER_CENTER_Z],
                            [DRAWER_CENTER_X - 0.001, DRAWER_CENTER_Y + 0.001, DRAWER_CENTER_Z],
                        ],
                        "validate_goal_rectangle_corners": False,
                        "articulation_config": None,
                        "subtask_config": {"obj_goal_thresh": 0.5},
                    },
                    # Close container (hands free again!)
                    {"type": "navigate", "target": articulation},
                    {
                        "type": "close",
                        "articulation_type":                    ARTICULATIONS[articulation]["articulation_type"],
                        "articulation_id":                      ARTICULATIONS[articulation]["articulation_id"],
                        "articulation_handle_link_idx":         ARTICULATIONS[articulation]["articulation_handle_link_idx"],
                        "articulation_handle_active_joint_idx": ARTICULATIONS[articulation]["articulation_handle_active_joint_idx"],
                        "articulation_relative_handle_pos":     ARTICULATIONS[articulation]["articulation_relative_handle_pos"],
                    },
                ]
        
        return None
    
    # -----------------------------------------------------------------------
    # Handler: swap two objects
    # -----------------------------------------------------------------------
    def _detect_and_handle_swap(self, command):
        command_lower = command.lower()
        
        if not any(word in command_lower for word in ["swap", "switch", "exchange"]):
            return None
        
        obj_ids = []
        for friendly_name in FRIENDLY_NAME_TO_OBJ_ID.keys():
            if friendly_name in command:
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
    
    # -----------------------------------------------------------------------
    # LLM fallback helpers
    # -----------------------------------------------------------------------
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
                "accessible": self.object_locations[obj_id] not in ["drawer", "fridge"]
            }
        
        return context
    
    def _build_system_prompt(self, scene_context):
        accessible_objs = []
        inaccessible_objs = []
        
        for obj_name, obj_data in scene_context["objects"].items():
            if obj_data["accessible"]:
                accessible_objs.append(f"  - {obj_name} (on counter - directly accessible)")
            else:
                inaccessible_objs.append(f"  - {obj_name} (INSIDE {obj_data['current_location'].upper()} - must open first!)")
        
        accessible_list = "\n".join(accessible_objs)
        inaccessible_list = "\n".join(inaccessible_objs)
        
        prompt = f"""You are a robot task planner. Convert user commands into subtask sequences.

Accessible objects:
{accessible_list}

Inaccessible objects (inside containers):
{inaccessible_list}

Available actions: navigate, pick, place, open, close

RULES:
1. ALWAYS navigate before pick/place/open/close
2. Can only hold ONE object at a time
3. Cannot open/close while holding object
4. When placing on counter, specify exact goal_pos
5. Only close drawer/fridge AFTER placing object (not while holding!)

Respond ONLY with a JSON array."""
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
    
    def _translate_friendly_names(self, subtasks):
        for subtask in subtasks:
            if "obj_id" in subtask:
                friendly = subtask["obj_id"]
                if friendly in FRIENDLY_NAME_TO_OBJ_ID:
                    subtask["obj_id"] = FRIENDLY_NAME_TO_OBJ_ID[friendly]
        return subtasks
    
    def update_object_location(self, obj_id, new_location, new_position=None):
        if obj_id in self.object_locations:
            self.object_locations[obj_id] = new_location
        if new_position and obj_id in self.object_positions:
            self.object_positions[obj_id] = new_position
