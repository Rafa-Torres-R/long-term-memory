"""
Live Session Command Parser
Handles natural language → subtask list conversion for the live interactive session.

Handler priority order:
  1. swap         — "swap bowl_0 and bowl_1"
  2. pick+place → container  — "pick apple and put in drawer"
  3. pick+place → counter    — "pick apple and place on counter"
  4. pick from container     — "pick up the apple" (apple is in fridge/drawer)
  5. pick from counter       — "pick up bowl 0"
  6. place only              — "place bowl 3 on counter" (robot already holding it)
  7. LLM fallback            — anything else
"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# scene_config lives in interactive_robot/ — make sure it's on sys.path
from scene_config import (
    OBJECTS, ARTICULATIONS, FRIENDLY_NAME_TO_OBJ_ID,
    COUNTER_X, COUNTER_Y, COUNTER_Z,
    DRAWER_CENTER_X, DRAWER_CENTER_Y, DRAWER_CENTER_Z,
)

# Fridge interior placement coords
FRIDGE_CENTER = [-2.09654, 1.28399, -1.01871]
COUNTER_POS   = [COUNTER_X, COUNTER_Y, COUNTER_Z]
DRAWER_POS    = [DRAWER_CENTER_X, DRAWER_CENTER_Y, DRAWER_CENTER_Z]

# Which locations require opening before access
CONTAINER_LOCATIONS = {"drawer", "fridge"}


def _corners(pos, delta=0.001):
    """Generate tight goal_rectangle_corners around a position."""
    x, y, z = pos
    return [
        [x + delta, y - delta, z],
        [x + delta, y + delta, z],
        [x - delta, y - delta, z],
        [x - delta, y + delta, z],
    ]


def _place_subtask(obj_id, pos):
    """Build a fully-specified place subtask dict."""
    return {
        "type": "place",
        "obj_id": obj_id,
        "goal_pos": pos,
        "goal_rectangle_corners": _corners(pos),
        "validate_goal_rectangle_corners": False,
        "articulation_config": None,
    }


def _open_subtask(articulation):
    """Build an open subtask dict for the given articulation name."""
    cfg = ARTICULATIONS[articulation]
    return {
        "type": "open",
        "articulation_type":                    cfg["articulation_type"],
        "articulation_id":                      cfg["articulation_id"],
        "articulation_handle_link_idx":         cfg["articulation_handle_link_idx"],
        "articulation_handle_active_joint_idx": cfg["articulation_handle_active_joint_idx"],
        "articulation_relative_handle_pos":     cfg["articulation_relative_handle_pos"],
        "obj_id": None,  # filled in by caller if needed
    }


def _close_subtask(articulation):
    """Build a close subtask dict for the given articulation name."""
    cfg = ARTICULATIONS[articulation]
    return {
        "type": "close",
        "articulation_type":                    cfg["articulation_type"],
        "articulation_id":                      cfg["articulation_id"],
        "articulation_handle_link_idx":         cfg["articulation_handle_link_idx"],
        "articulation_handle_active_joint_idx": cfg["articulation_handle_active_joint_idx"],
        "articulation_relative_handle_pos":     cfg["articulation_relative_handle_pos"],
    }


def _placement_pos(location):
    """Return the world-space placement position for a given location string."""
    if location == "fridge":
        return FRIDGE_CENTER
    if location == "drawer":
        return DRAWER_POS
    return COUNTER_POS  # counter / anywhere else


class LiveCommandParser:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Loading Qwen model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print(f"Model loaded on: {self.model.device}")

        # Track where each object currently is — updated after every command
        self.object_locations = {
            obj_id: info["initial_location"]
            for obj_id, info in OBJECTS.items()
        }
        self.object_positions = {
            obj_id: info["initial_position"]
            for obj_id, info in OBJECTS.items()
        }

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def parse_command(self, user_command):
        print(f"  🔍 object_locations: {self.object_locations}")
        cmd = user_command.lower().replace("_", " ")

        # 1. Swap two objects
        result = self._handle_swap(cmd)
        if result is not None:
            print("🔧 Using: swap handler")
            return result

        # 2. Pick AND place INTO container (drawer/fridge)
        result = self._handle_pick_and_place_into_container(cmd)
        if result is not None:
            print("🔧 Using: pick+place→container handler")
            return result

        # 3. Pick AND place ONTO counter/table/surface
        result = self._handle_pick_and_place_onto_counter(cmd)
        if result is not None:
            print("🔧 Using: pick+place→counter handler")
            return result

        # 4. Pick only — object is inside a container
        result = self._handle_pick_from_container(cmd)
        if result is not None:
            print("🔧 Using: container pick handler")
            return result

        # 5. Pick only — object is on the counter
        result = self._handle_pick_from_counter(cmd)
        if result is not None:
            print("🔧 Using: counter pick handler")
            return result

        # 6. Place only — robot is already holding the object
        result = self._handle_place_only(cmd)
        if result is not None:
            print("🔧 Using: place-only handler")
            return result

        # 7. LLM fallback
        print("🔧 Using: LLM fallback")
        return self._llm_fallback(user_command)

    def update_object_location(self, obj_id, new_location, new_position=None):
        """Call this after every successfully executed command."""
        if obj_id in self.object_locations:
            self.object_locations[obj_id] = new_location
        if new_position is not None and obj_id in self.object_positions:
            self.object_positions[obj_id] = new_position

    # -----------------------------------------------------------------------
    # Handler 1: Swap
    # -----------------------------------------------------------------------
    def _handle_swap(self, cmd):
        if not any(w in cmd for w in ["swap", "switch", "exchange"]):
            return None

        matched = [fn for fn in FRIENDLY_NAME_TO_OBJ_ID if fn.replace("_", " ") in cmd]
        if len(matched) != 2:
            return None

        fn_a, fn_b  = matched
        id_a, id_b  = FRIENDLY_NAME_TO_OBJ_ID[fn_a], FRIENDLY_NAME_TO_OBJ_ID[fn_b]
        pos_a, pos_b = self.object_positions[id_a], self.object_positions[id_b]
        if not pos_a or not pos_b:
            return None

        staging = [0.0, -2.0, 1.05]
        return [
            {"type": "navigate", "target": fn_a.replace("_", " ")},
            {"type": "pick",     "obj_id": id_a},
            {"type": "navigate", "target": "staging"},
            _place_subtask(id_a, staging),
            {"type": "navigate", "target": fn_b.replace("_", " ")},
            {"type": "pick",     "obj_id": id_b},
            {"type": "navigate", "target": "counter"},
            _place_subtask(id_b, pos_a),
            {"type": "navigate", "target": "staging"},
            {"type": "pick",     "obj_id": id_a},
            {"type": "navigate", "target": "counter"},
            _place_subtask(id_a, pos_b),
        ]

    # -----------------------------------------------------------------------
    # Handler 2: Pick AND place INTO container
    # e.g. "pick apple and put it in the fridge"
    # -----------------------------------------------------------------------
    def _handle_pick_and_place_into_container(self, cmd):
        has_pick  = any(w in cmd for w in ["pick", "get", "grab", "take"])
        has_place = any(w in cmd for w in ["put", "place", "set", "store"])
        dest      = "drawer" if "drawer" in cmd else ("fridge" if "fridge" in cmd else None)
        if not (has_pick and has_place and dest):
            return None

        obj_id, fn = self._find_object_in_cmd(cmd)
        if obj_id is None:
            return None

        current_loc = self.object_locations[obj_id]
        subtasks = []

        # If object itself is currently in a DIFFERENT container, open that first
        if current_loc in CONTAINER_LOCATIONS and current_loc != dest:
            o = _open_subtask(current_loc)
            o["obj_id"] = obj_id
            subtasks += [
                {"type": "navigate", "target": current_loc},
                o,
            ]

        # Pick the object
        subtasks += [
            {"type": "navigate", "target": fn},
            {"type": "pick",     "obj_id": obj_id},
        ]

        # Open destination container, place inside, close
        dest_pos = _placement_pos(dest)
        o = _open_subtask(dest)
        o["obj_id"] = obj_id
        subtasks += [
            {"type": "navigate", "target": dest},
            o,
            {"type": "navigate", "target": dest},
            _place_subtask(obj_id, dest_pos),
            {"type": "navigate", "target": dest},
            _close_subtask(dest),
        ]

        # Close original container if it was different
        if current_loc in CONTAINER_LOCATIONS and current_loc != dest:
            subtasks += [
                {"type": "navigate", "target": current_loc},
                _close_subtask(current_loc),
            ]

        return subtasks

    # -----------------------------------------------------------------------
    # Handler 3: Pick AND place ONTO counter/table
    # e.g. "pick bowl 3 and place on counter"
    # -----------------------------------------------------------------------
    def _handle_pick_and_place_onto_counter(self, cmd):
        has_pick    = any(w in cmd for w in ["pick", "get", "grab", "take"])
        has_place   = any(w in cmd for w in ["put", "place", "set"])
        has_counter = any(w in cmd for w in ["counter", "table", "surface"])
        if not (has_pick and has_place and has_counter):
            return None

        obj_id, fn = self._find_object_in_cmd(cmd)
        if obj_id is None:
            return None

        current_loc = self.object_locations[obj_id]
        subtasks = []

        # Open container if needed
        if current_loc in CONTAINER_LOCATIONS:
            o = _open_subtask(current_loc)
            o["obj_id"] = obj_id
            subtasks += [
                {"type": "navigate", "target": current_loc},
                o,
            ]

        # Pick
        subtasks += [
            {"type": "navigate", "target": fn},
            {"type": "pick",     "obj_id": obj_id},
        ]

        # Place on counter
        subtasks += [
            {"type": "navigate", "target": "counter"},
            _place_subtask(obj_id, COUNTER_POS),
        ]

        # Close container if we opened it
        if current_loc in CONTAINER_LOCATIONS:
            subtasks += [
                {"type": "navigate", "target": current_loc},
                _close_subtask(current_loc),
            ]

        return subtasks

    # -----------------------------------------------------------------------
    # Handler 4: Pick only — object is in a container
    # e.g. "pick up the apple"  (apple is in fridge)
    # -----------------------------------------------------------------------
    def _handle_pick_from_container(self, cmd):
        if not any(w in cmd for w in ["pick", "get", "grab", "take"]):
            return None

        obj_id, fn = self._find_object_in_cmd(cmd)
        if obj_id is None:
            return None

        location = self.object_locations[obj_id]
        if location not in CONTAINER_LOCATIONS:
            return None

        o = _open_subtask(location)
        o["obj_id"] = obj_id

        return [
            {"type": "navigate", "target": location},
            o,
            {"type": "navigate", "target": fn},
            {"type": "pick",     "obj_id": obj_id},
            {"type": "navigate", "target": location},
            _close_subtask(location),
        ]

    # -----------------------------------------------------------------------
    # Handler 5: Pick only — object is on the counter
    # e.g. "pick up bowl 0"
    # -----------------------------------------------------------------------
    def _handle_pick_from_counter(self, cmd):
        if not any(w in cmd for w in ["pick", "get", "grab", "take"]):
            return None

        obj_id, fn = self._find_object_in_cmd(cmd)
        if obj_id is None:
            return None

        location = self.object_locations[obj_id]
        if location in CONTAINER_LOCATIONS:
            return None  # let handler 4 deal with it

        return [
            {"type": "navigate", "target": fn},
            {"type": "pick",     "obj_id": obj_id},
        ]

    # -----------------------------------------------------------------------
    # Handler 6: Place only — robot already holding the object
    # e.g. "place it on the counter" / "put it in the fridge"
    # -----------------------------------------------------------------------
    def _handle_place_only(self, cmd):
        has_place = any(w in cmd for w in ["place", "put", "set", "drop"])
        if not has_place:
            return None

        # Destination
        if "drawer" in cmd:
            dest = "drawer"
        elif "fridge" in cmd:
            dest = "fridge"
        elif any(w in cmd for w in ["counter", "table", "surface"]):
            dest = "counter"
        else:
            return None

        # Find object — either named or currently held
        obj_id, fn = self._find_object_in_cmd(cmd)
        if obj_id is None:
            # Fall back to whatever is currently held
            for oid, loc in self.object_locations.items():
                if loc == "held":
                    obj_id = oid
                    break
        if obj_id is None:
            return None

        dest_pos = _placement_pos(dest)

        if dest in CONTAINER_LOCATIONS:
            o = _open_subtask(dest)
            o["obj_id"] = obj_id
            return [
                {"type": "navigate", "target": dest},
                o,
                {"type": "navigate", "target": dest},
                _place_subtask(obj_id, dest_pos),
                {"type": "navigate", "target": dest},
                _close_subtask(dest),
            ]
        else:
            return [
                {"type": "navigate", "target": "counter"},
                _place_subtask(obj_id, dest_pos),
            ]

    # -----------------------------------------------------------------------
    # Handler 7: LLM fallback
    # -----------------------------------------------------------------------
    def _llm_fallback(self, user_command):
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Command: {user_command}"},
        ]
        text   = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
        generated = outputs[0][inputs.input_ids.shape[-1]:]
        response  = self.tokenizer.decode(generated, skip_special_tokens=True)
        subtasks  = self._extract_json(response)
        return self._translate_friendly_names(subtasks)

    def _build_system_prompt(self):
        accessible   = []
        inaccessible = []
        for obj_id, info in OBJECTS.items():
            fn  = info["friendly_name"]
            loc = self.object_locations[obj_id]
            if loc in CONTAINER_LOCATIONS:
                inaccessible.append(f"  - {fn} (inside {loc.upper()} — must open first)")
            else:
                accessible.append(f"  - {fn} (on counter)")

        return f"""You are a robot task planner. Convert user commands into subtask JSON arrays.
Accessible objects:
{chr(10).join(accessible)}
Inaccessible objects:
{chr(10).join(inaccessible)}
Available subtask types: navigate, pick, place, open, close
RULES:
1. Always navigate before pick/place/open/close
2. Robot can hold only ONE object at a time
3. Cannot open/close while holding an object
4. Always close containers after picking or placing inside
5. place subtasks must include goal_pos as [x, y, z]
Counter placement position: {COUNTER_POS}
Fridge placement position:  {FRIDGE_CENTER}
Drawer placement position:  {DRAWER_POS}
Example — pick from fridge:
[
  {{"type":"navigate","target":"fridge"}},
  {{"type":"open","articulation":"fridge","obj_id":"013_apple-0"}},
  {{"type":"navigate","target":"apple"}},
  {{"type":"pick","obj_id":"013_apple-0"}},
  {{"type":"navigate","target":"fridge"}},
  {{"type":"close","articulation":"fridge"}}
]
Example — place on counter:
[
  {{"type":"navigate","target":"counter"}},
  {{"type":"place","obj_id":"013_apple-0","goal_pos":{COUNTER_POS}}}
]
Respond ONLY with a JSON array, no other text."""

    def _extract_json(self, response):
        try:
            start = response.find('[')
            end   = response.rfind(']') + 1
            if start != -1 and end > 0:
                return json.loads(response[start:end])
        except Exception as e:
            print(f"  ⚠️  JSON parse error: {e}")
        return []

    def _translate_friendly_names(self, subtasks):
        """Replace any friendly name used as obj_id with the real object ID."""
        for subtask in subtasks:
            if "obj_id" in subtask and subtask["obj_id"] in FRIENDLY_NAME_TO_OBJ_ID:
                subtask["obj_id"] = FRIENDLY_NAME_TO_OBJ_ID[subtask["obj_id"]]
        return subtasks

    # -----------------------------------------------------------------------
    # Helper: find the first mentioned object in a command string
    # Returns (obj_id, friendly_name_normalized) or (None, None)
    # -----------------------------------------------------------------------
    def _find_object_in_cmd(self, cmd):
        for friendly_name, obj_id in FRIENDLY_NAME_TO_OBJ_ID.items():
            normalized = friendly_name.replace("_", " ")
            if normalized in cmd:
                return obj_id, normalized
        return None, None
