#!/usr/bin/env python3
import os
import sys
import bz2
import urllib.parse
import capnp
import warnings
from hashlib import sha256
import cv2
import numpy as np
from cereal import log as capnp_log
from openpilot.tools.lib.logreader import LogReader

import codecs
import av

def hash_256(link):
  hsh = str(sha256((link.split("?")[0]).encode('utf-8')).hexdigest())
  return hsh

def com2dir(com):
  if com[0] == 1 and com[1] == 0:
     return 'F'
  elif com[0] == 0 and com[1] == +1:
     return 'L'
  elif com[0] == 0 and com[1] == -1:
     return 'R'
  elif com[0] == 1 and com[1] == -1:
     return 'FR'
  elif com[0] == 1 and com[1] == +1:
     return 'FL'
  elif com[-1] == -1 and com[0] == 0:
     return 'B'
  else:
     print("#######", com)
  

# capnproto <= 0.8.0 throws errors converting byte data to string
# below line catches those errors and replaces the bytes with \x__

video = os.path.join('./training_data', sys.argv[1], 'ecamera.hevc')
#video = os.path.join(*log_path.split("/")[:-1], 'fcamera.hevc')
lr = LogReader('./training_data/' + sys.argv[1] + '/rlog', sort_by_time=True)
cnt = 100
frameId = 0

dirs = os.path.join('./', 'dataset', hash_256(sys.argv[1])[:8])
os.makedirs(dirs, exist_ok=True)
storage = {}

for msg in lr:
    if "wideRoadCameraState" in str(msg) and 'frameId' in str(msg):
        #print(msg)
        frameId = msg.wideRoadCameraState.frameId
    if "testJoystick" in str(msg) and 'axes' in str(msg):
        storage[int(frameId)] = msg.testJoystick.axes

container = av.open(video)
for frame in container.decode(video=0):
    if frame.index in storage.keys() and com2dir(storage[frame.index]) != None:
        img_yuv = frame.to_ndarray(format=av.video.format.VideoFormat('yuv420p'))
        h, w = int(img_yuv.shape[0]*2/3), int(img_yuv.shape[1])
        img_yuv = img_yuv[:h,:w]
        img_yuv = img_yuv[:h, 280:-440]
        print(frame.index, img_yuv.shape, storage[frame.index])
        cv2.imwrite(dirs + '/' + str(frame.index) + '_' + com2dir(storage[frame.index]) + '.png', img_yuv.astype(np.uint8))

