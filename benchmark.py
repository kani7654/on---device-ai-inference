import onnxruntime as ort
import numpy as np
import time

def benchmark(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, 1, 28, 28).astype(np.float32)

    start = time.time()
    for _ in range(50):
        session.run(None, {input_name: dummy_input})
    end = time.time()

    print(f"{model_path} Avg Latency: {(end-start)/50:.4f} sec")

benchmark("model.onnx")
benchmark("model_int8.onnx")
