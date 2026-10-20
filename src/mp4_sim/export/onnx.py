"""Actor-to-ONNX conversion.

Legacy torch exporter (``dynamo=False``), opset 18, fixed shapes, input
``obs`` [1, 45], output ``actions`` [1, 12], normalization inside the graph.
"""
