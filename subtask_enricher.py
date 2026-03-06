"""
Subtask Enricher: Converts simplified LLM output to full ManiSkill-HAB format
"""
import json
from scene_config import ARTICULATIONS, OBJECTS, OBJECT_OFFSETS

class SubtaskEnricher:
    def __init__(self):
        self.articulations = ARTICULATIONS
        self.objects = OBJECTS
        self.object_placement_counter = {}
        self.uid_counter = 0
    
    def enrich_subtasks(self, simplified_subtasks):
        enriched = []
        
        for subtask in simplified_subtasks:
            task_type = subtask["type"]
            
            if task_type == "navigate":
                enriched.append(self._enrich_navigate(subtask))
            elif task_type == "pick":
                enriched.append(self._enrich_pick(subtask))
            elif task_type == "place":
                enriched.append(self._enrich_place(subtask))
            elif task_type == "open":
                enriched.append(self._enrich_open(subtask))
            elif task_type == "close":
                enriched.append(self._enrich_close(subtask))
        
        return enriched
    
    def _generate_uid(self, prefix):
        self.uid_counter += 1
        return f"{prefix}_{self.uid_counter}"
    
    def _enrich_navigate(self, subtask):
        return {
            "type": "navigate",
            "uid": self._generate_uid("nav")
        }
    
    def _enrich_pick(self, subtask):
        obj_id = subtask["obj_id"]
        from_location = subtask.get("from", None)
        
        enriched = {
            "type": "pick",
            "uid": self._generate_uid("pick"),
            "obj_id": obj_id,
        }
        
        if from_location and from_location in self.articulations:
            art_config = self.articulations[from_location]
            enriched["articulation_config"] = {
                "articulation_type": art_config["articulation_type"],
                "articulation_id": art_config["articulation_id"],
                "articulation_handle_link_idx": art_config["articulation_handle_link_idx"],
                "articulation_handle_active_joint_idx": art_config["articulation_handle_active_joint_idx"],
            }
        else:
            enriched["articulation_config"] = None
        
        return enriched
    
    def _enrich_place(self, subtask):
        obj_id = subtask["obj_id"]
        target = subtask.get("target", "counter")
        
        enriched = {
            "type": "place",
            "uid": self._generate_uid("place"),
            "obj_id": obj_id,
        }
        
        if target == "drawer" and "drawer" in self.articulations:
            art_config = self.articulations["drawer"]
            
            if target not in self.object_placement_counter:
                self.object_placement_counter[target] = 0
            
            offset_idx = self.object_placement_counter[target]
            self.object_placement_counter[target] += 1
            
            x_offset, y_offset = OBJECT_OFFSETS.get(offset_idx, [0.0, 0.0])
            center = art_config["placement_coords"]["center"]
            
            goal_x = center[0] + x_offset
            goal_y = center[1] + y_offset
            goal_z = center[2]
            
            enriched["goal_pos"] = [goal_x, goal_y, goal_z]
            enriched["goal_rectangle_corners"] = [
                [goal_x - 0.0015, goal_y + 0.001, goal_z],
                [goal_x + 0.0015, goal_y - 0.001, goal_z],
                [goal_x - 0.0015, goal_y + 0.001, goal_z],
                [goal_x + 0.0015, goal_y - 0.001, goal_z]
            ]
            enriched["validate_goal_rectangle_corners"] = False
            enriched["articulation_config"] = {
                "articulation_type": art_config["articulation_type"],
                "articulation_id": art_config["articulation_id"],
                "articulation_handle_link_idx": art_config["articulation_handle_link_idx"],
                "articulation_handle_active_joint_idx": art_config["articulation_handle_active_joint_idx"],
            }
        elif "goal_pos" in subtask:
            goal_pos = subtask["goal_pos"]
            enriched["goal_pos"] = goal_pos
            enriched["goal_rectangle_corners"] = [
                [goal_pos[0] - 0.0015, goal_pos[1] + 0.001, goal_pos[2]],
                [goal_pos[0] + 0.0015, goal_pos[1] - 0.001, goal_pos[2]],
                [goal_pos[0] - 0.0015, goal_pos[1] + 0.001, goal_pos[2]],
                [goal_pos[0] + 0.0015, goal_pos[1] - 0.001, goal_pos[2]]
            ]
            enriched["validate_goal_rectangle_corners"] = False
            enriched["articulation_config"] = None
        else:
            enriched["goal_pos"] = [0.0, 0.0, 1.0]
            enriched["goal_rectangle_corners"] = []
            enriched["validate_goal_rectangle_corners"] = False
            enriched["articulation_config"] = None
        
        return enriched
    
    def _enrich_open(self, subtask):
        articulation_name = subtask.get("articulation", "drawer")
        obj_id = subtask.get("obj_id", None)  # May be None
        
        if articulation_name not in self.articulations:
            raise ValueError(f"Unknown articulation: {articulation_name}")
        
        art_config = self.articulations[articulation_name]
        
        enriched = {
            "type": "open",
            "uid": self._generate_uid("open"),
            "articulation_type": art_config["articulation_type"],
            "articulation_id": art_config["articulation_id"],
            "articulation_handle_link_idx": art_config["articulation_handle_link_idx"],
            "articulation_handle_active_joint_idx": art_config["articulation_handle_active_joint_idx"],
            "articulation_relative_handle_pos": art_config["articulation_relative_handle_pos"],
        }
        
        # ⭐ FIX: Only include obj_id if it's not None
        if obj_id is not None:
            enriched["obj_id"] = obj_id
        
        return enriched
    
    def _enrich_close(self, subtask):
        articulation_name = subtask.get("articulation", "drawer")
        
        if articulation_name not in self.articulations:
            raise ValueError(f"Unknown articulation: {articulation_name}")
        
        art_config = self.articulations[articulation_name]
        
        return {
            "type": "close",
            "uid": self._generate_uid("close"),
            "articulation_type": art_config["articulation_type"],
            "articulation_id": art_config["articulation_id"],
            "articulation_handle_link_idx": art_config["articulation_handle_link_idx"],
            "articulation_handle_active_joint_idx": art_config["articulation_handle_active_joint_idx"],
            "articulation_relative_handle_pos": art_config["articulation_relative_handle_pos"],
            "remove_obj_id": None
        }
