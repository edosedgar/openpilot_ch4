#!/usr/bin/env python3
import cv2
import time
import numpy as np
import os
import argparse
import onnx
from pathlib import Path

os.environ["ZMQ"] = "1"
from cereal import messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from openpilot.selfdrive.modeld.runners import ModelRunner
import onnxruntime as ort

INPUT_SHAPE = (240, 240, 3)

CLASS_NAME_TO_CMD = {
    'F': {'x': 1.0, 'y': 0.0},
    'R': {'x': 0.0, 'y': -1.0},
    'L': {'x': 0.0, 'y': 1.0},
    'FL': {'x': 1.0, 'y': 1.0},
    'FR': {'x': 1.0, 'y': -1.0},
}

class KlayRunner:
  def __init__(self, onnx_path: str):
    self.model = onnx.load(onnx_path)
    self.class_names = list(CLASS_NAME_TO_CMD.keys())
    onnx.checker.check_model(self.model)
    self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

  def preprocess_image(self, img):
    img = cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))

    # Convert from BGR to RGB (since OpenCV loads images in BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0  # Convert values to [0,1]
    img = img.transpose(2, 0, 1)  # HWC -> CHW

    # Normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img - mean) / std

    # Add batch dimension
    img = np.expand_dims(img, 0).astype(np.float32)

    return img

  def run(self, img):
    img = self.preprocess_image(img)
    input_name = self.session.get_inputs()[0].name
    raw_result = self.session.run(None, {input_name: img})

    output_probabilities = raw_result[0][0]  # adjust based on your model's output structure
    predicted_class = np.argmax(output_probabilities)
    return [{
        "pred_class": self.class_names[predicted_class],
        "prob": output_probabilities[predicted_class]
    }]

  def draw_boxes(self, img, objects):
    img = cv2.resize(img, INPUT_SHAPE)
    for obj in objects:
      pt1 = obj['pt1']
      pt2 = tuple(obj['pt2'])
      cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
      cv2.putText(img, f"{obj['pred_class']} {obj['prob']:.2f}", pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img


def main(debug=False, model_path=None):
  klay_runner = KlayRunner(model_path)
  pm = messaging.PubMaster(['customReservedRawData1'])
  del os.environ["ZMQ"]
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

    pm.send('customReservedRawData1', msg)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Receive VisionIPC frames, run klay and publish outputs")
  parser.add_argument('--model-path', help='path of model')
  parser.add_argument("--debug", action="store_true", help="debug output")
  args = parser.parse_args()

  main(debug=args.debug, model_path=args.model_path)
