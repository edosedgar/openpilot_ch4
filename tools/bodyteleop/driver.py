#!/usr/bin/env python3
import json
import logging
import time
import os

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
  msg = messaging.new_message()
  msg.customReservedRawData1 = json.dumps(ctrls[:10]).encode()
  print(msg.customReservedRawData1)
  pm.send('customReservedRawData1', msg)

if __name__ == "__main__":
  main()
