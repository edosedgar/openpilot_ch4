#!/usr/bin/env python3

from dataclasses import dataclass
import json
import logging
import time
from typing import Dict, List, Set

from cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.tools.bodyteleop.constants import ControlData

@dataclass
class ControlState:
  past_executions: Set[int]
  exec_id: int
  controls_list: List[Dict[str, float]]
  control_idx: int

TIME_GAP_THRESHOLD = 0.5
last_control_send_time = time.monotonic()
logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

C1_FILE = '/tmp/c1.log'

def send_control_message(pm, x, y, source):
  global last_control_send_time
  if not engaged and source == 'model':
    return

  msg = messaging.new_message('testJoystick')
  msg.testJoystick.axes = [x, y]
  msg.testJoystick.buttons = [False]
  pm.send('testJoystick', msg)
  if source != 'dummy':
    logger.info(f"bodycontrol|{source} (x, y): ({x}, {y})")
  last_control_send_time = time.monotonic()

def update_state(state: ControlState, data: ControlData) -> None:
  state.past_executions.add(data.exec_id)
  state.exec_id = data.exec_id
  state.controls_list = data.controls_list
  state.control_idx = 0

def print_state(state: ControlState) -> None:
  print(f'exec_id: {state.exec_id}')
  #print(f'past execs: {state.past_executions}')
  print(f'controls_list: {state.control_idx} / {len(state.controls_list)}')

def execute_control(state: ControlState, pm: messaging.PubMaster) -> None:
  if state.control_idx >= len(state.controls_list):
    send_control_message(pm, 0.0, 0.0, 'dummy')
    return

  controls = state.controls_list[state.control_idx]
  send_control_message(pm, controls['x'], controls['y'], 'model')
  state.control_idx += 1
  print_state(state)

def main():
  rk = Ratekeeper(20.0)
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['customReservedRawData0', 'customReservedRawData1'])
  cycle = 0
  state = ControlState(past_executions=set(), exec_id=0, controls_list=[], control_idx=0)

  global engaged
  engaged = True
  while True:
    sm.update(0)
    cycle += 1

    if sm.updated['customReservedRawData0']:
      controls = json.loads(sm['customReservedRawData0'].decode())
      if controls['x'] == 0.5:
        engaged = False
        print('disengaged')
      elif controls['y'] == 0.5:
        engaged = True
        print('engaged')
      else:
        send_control_message(pm, -controls['x'], controls['y'], 'wasd')
    elif sm.updated['customReservedRawData1']:
      data = json.loads(sm['customReservedRawData1'].decode())
      control_data = ControlData(**data)
      if control_data.exec_id not in state.past_executions:
        update_state(state, control_data)
        print(f'Received {control_data.exec_id}: {control_data.controls_list} at cycle {cycle} time {time.monotonic()}')

    now = time.monotonic()
    if now > last_control_send_time + TIME_GAP_THRESHOLD:
      execute_control(state, pm)

    rk.keep_time()

if __name__ == "__main__":
  main()
