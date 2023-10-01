import torch
import torch.nn as nn
from torchvision import models
import argparse

def load_model(model_path):
    # Define the model structure (architecture) as it was during training
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'], strict=True)
    print("loaded success")
    return model

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument("model_path", type=str, help="Path to the PyTorch .pth model file")
    args = parser.parse_args()

    # Load the PyTorch model
    model = load_model(args.model_path)
    model.eval()  # Set the model to evaluation mode

    # Convert to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)  # example input tensor of the shape the model expects
    onnx_path = args.model_path.replace(".pth", ".onnx").replace('checkpoints', 'onnx')  # Save ONNX model with the same name, but .onnx extension
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
    
    print(f"Model exported to {onnx_path}")