#!/usr/bin/env python3
#import cv2
import cv2
import time
import numpy as np
import argparse
import json
from pathlib import Path

from cereal import messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime

INPUT_SHAPE = (224, 224, 3)

OUTPUT_SHAPE = (1, 4)
MODEL_PATHS = {
  ModelRunner.THNEED: Path(__file__).parent / 'models/klay.thneed',
  ModelRunner.ONNX: Path(__file__).parent / 'models/klay.onnx'}


CLASS_NAME_TO_CMD = {
    'F': {'x': 1.0, 'y': 0.0},
    'R': {'x': 0.0, 'y': -1.0},
    'L': {'x': 0.0, 'y': 1.0},
    'B': {'x': -1.0, 'y': 0.0},
}

class KlayRunner:
  def __init__(self):
    self.class_names = list(CLASS_NAME_TO_CMD.keys())
    self.output = np.zeros(np.prod(OUTPUT_SHAPE), dtype=np.float32)
    self.model = ModelRunner(MODEL_PATHS, self.output, Runtime.CPU, False, None)
    self.model.addInput("input.1", None)

  def preprocess_image(self, img):
    img = img[:img.shape[0], 280:-440]
    img = cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]), interpolation=cv2.INTER_AREA)

    img = img / 255.0  # Convert values to [0,1]
    img = img.transpose(2, 0, 1)  # HWC -> CHW

    # Add batch dimension
    img = np.expand_dims(img, 0).astype(np.float32)

    return img

  def run(self, img):
    img = self.preprocess_image(img)
    self.model.setInputBuffer("input.1", img.flatten())
    self.model.execute()
    raw_result = self.output
    print(raw_result)
    output_probabilities = raw_result  # adjust based on your model's output structure
    predicted_class = np.argmax(output_probabilities)
    return [{
        "pred_class": self.class_names[predicted_class],
        "prob": output_probabilities[predicted_class]
    }]


def main(debug=False):
  klay_runner = KlayRunner()
  pm = messaging.PubMaster(['customReservedRawData1'])
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)

  while not vipc_client.connect(False):
    time.sleep(0.1)

  while True:
    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any():
      continue

    imgff = yuv_img_raw.data.reshape(-1, vipc_client.stride)
    imgff = imgff[:vipc_client.height, :vipc_client.width]
    img = np.stack([imgff, imgff, imgff], axis=-1)
    outputs = klay_runner.run(img)
    print(outputs)
    pred = outputs[0]['pred_class']

    cmd = CLASS_NAME_TO_CMD[pred]

    msg = messaging.new_message()
    msg.customReservedRawData1 = json.dumps(cmd).encode()
    pm.send('customReservedRawData1', msg)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Receive VisionIPC frames, run KLAY and publish outputs")
  parser.add_argument("--debug", action="store_true", help="debug output")
  args = parser.parse_args()

  main(debug=args.debug)
