#!/usr/bin/env python3
from dataclasses import asdict
import json
import logging
import time
import os
from typing import List
from openpilot.common.realtime import Ratekeeper
from openpilot.tools.bodyteleop.constants import ControlData
from openpilot.tools.bodyteleop.constants import ControlList

os.environ["ZMQ"] = "1"
from cereal import messaging

TIME_GAP_THRESHOLD = 0.5
last_control_send_time = time.monotonic()
logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

LOG_FILE = 'tools/bodyteleop/data/c1.log'

def forward(steps: int) -> ControlList:
  return [{'x': 1.0, 'y': 0.0} for _ in range(steps)]

def left(steps: int) -> ControlList:
  return [{'x': 0.0, 'y': 1.0} for _ in range(steps)]

def right(steps: int) -> ControlList:
  return [{'x': 0.0, 'y': -1.0} for _ in range(steps)]

def main():
  pm = messaging.PubMaster(['customReservedRawData1'])

  print('starting')

  with open(LOG_FILE, 'r') as log:
    ctrls = [json.loads(line) for line in log.readlines()]

  print(f'Loaded {len(ctrls)} controls')
  exec_id = 0
  rk = Ratekeeper(0.75)

  ts = 0

  plan: List[ControlList] = [
    right(1),
  ] * 100

  start_time = int(time.time())
  exec_id = start_time
  while True:
    plan_idx = exec_id - start_time
    if plan_idx == len(plan):
      break

    data = ControlData(exec_id=exec_id, controls_list=plan[plan_idx])
    msg = messaging.new_message()
    msg.customReservedRawData1 = json.dumps(asdict(data)).encode()
    print(msg.customReservedRawData1)
    pm.send('customReservedRawData1', msg)

    rk.keep_time()
    ts += 1
    if ts >= len(plan[plan_idx]):
      ts = 0
      exec_id += 1

if __name__ == "__main__":
  main()
