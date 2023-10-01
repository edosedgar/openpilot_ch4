import argparse
import onnx
from onnx import helper
import numpy as np

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Modify ONNX model output shape")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model file")
    args = parser.parse_args()

    # Load the ONNX model
    model = onnx.load(args.model_path)

    # Get the name of the output tensor of the last node in the graph
    last_node_output_name = model.graph.node[-1].output[0]

    # Create a constant tensor for concatenation
    dummy_value_name = 'dummy_value'
    dummy_tensor = np.array([[0]], dtype=np.float32)
    dummy_value_tensor = helper.make_tensor(dummy_value_name, onnx.TensorProto.FLOAT, dummy_tensor.shape, dummy_tensor.flatten())
    dummy_value_node = helper.make_node('Constant', inputs=[], outputs=[dummy_value_name], value=dummy_value_tensor)

    # Create a Concat node to concatenate the original output and dummy tensor
    concat_output_name = 'concat_output'
    concat_node = helper.make_node(
        'Concat',
        inputs=[last_node_output_name, dummy_value_name],
        outputs=[concat_output_name],
        axis=1,  # Concatenate along the second axis
        name='ConcatOutput'
    )

    # Add the new nodes to the model's graph
    model.graph.node.extend([dummy_value_node, concat_node])

    # Update the model's output to be the output of the Concat node
    model.graph.output[0].name = concat_output_name

    # Optionally, check the model
    onnx.checker.check_model(model)

    # Save the modified ONNX model
    onnx.save(model, "/tmp/modified_model.onnx")
