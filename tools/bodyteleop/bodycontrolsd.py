#!/usr/bin/env python3
import json
import logging
import time

from cereal import messaging
from openpilot.common.realtime import Ratekeeper

TIME_GAP_THRESHOLD = 0.5
last_control_send_time = time.monotonic()
logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

LOG_FILE = '/tmp/joystick.log'
C1_FILE = '/tmp/c1.log'

def send_control_message(pm, x, y, source):
  global last_control_send_time
  msg = messaging.new_message('testJoystick')
  msg.testJoystick.axes = [x, y]
  msg.testJoystick.buttons = [False]
  pm.send('testJoystick', msg)
  logger.info(f"bodycontrol|{source} (x, y): ({x}, {y})")
  last_control_send_time = time.monotonic()

def main():
  rk = Ratekeeper(20.0)
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['customReservedRawData0', 'customReservedRawData1'])
  cycle = 0
  controls_list = []

  log = open(LOG_FILE, 'w+')

  while True:
    sm.update(0)
    cycle += 1

    if sm.updated['customReservedRawData0']:
      controls = json.loads(sm['customReservedRawData0'].decode())
      log.write(json.dumps(controls) + '\n')
      log.flush()
      send_control_message(pm, controls['x'], controls['y'], 'wasd')
    elif sm.updated['customReservedRawData1']:
      controls_list = json.loads(sm['customReservedRawData1'].decode())
      print(f'Received {controls_list} at cycle {cycle} time {time.monotonic()}')
      send_control_message(pm, controls['x'], controls['y'], 'wasd')
    else:
      now = time.monotonic()
      if now > last_control_send_time + TIME_GAP_THRESHOLD:
        print(f'cycle {cycle} time {time.monotonic()}')
        if len(controls_list) > 0:
          controls = controls_list.pop(0)
          send_control_message(pm, controls['x'], controls['y'], 'wasd')
        else:
          send_control_message(pm, 0.0, 0.0, 'dummy')

    rk.keep_time()

if __name__ == "__main__":
  main()
