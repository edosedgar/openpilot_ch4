#!/usr/bin/env python3
from dataclasses import asdict
import json
import logging
import time
import os
from openpilot.common.realtime import Ratekeeper
from openpilot.tools.bodyteleop.constants import ControlData

os.environ["ZMQ"] = "1"
from cereal import messaging

TIME_GAP_THRESHOLD = 0.5
last_control_send_time = time.monotonic()
logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

LOG_FILE = 'tools/bodyteleop/data/c1.log'


def main():
  pm = messaging.PubMaster(['customReservedRawData1'])

  print('starting')

  with open(LOG_FILE, 'r') as log:
    ctrls = [json.loads(line) for line in log.readlines()]

  print(f'Loaded {len(ctrls)} controls')
  exec_id = 0
  rk = Ratekeeper(20.0)

  while True:
    data = ControlData(exec_id=exec_id, controls_list=ctrls[:5])
    msg = messaging.new_message()
    msg.customReservedRawData1 = json.dumps(asdict(data)).encode()
    print(msg.customReservedRawData1)
    pm.send('customReservedRawData1', msg)

    rk.keep_time()

if __name__ == "__main__":
  main()
