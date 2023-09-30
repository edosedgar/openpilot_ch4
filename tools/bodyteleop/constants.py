from dataclasses import dataclass
from typing import Dict, List

ControlList = List[Dict[str, float]]

@dataclass
class ControlData():
  exec_id: int
  controls_list: ControlList