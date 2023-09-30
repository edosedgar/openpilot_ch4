from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ControlData():
  exec_id: int
  controls_list: List[Dict[str, float]]