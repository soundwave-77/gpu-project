import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import threading
import uvicorn
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


cuda.init()
app = FastAPI()


class LinearRegressionRequest(BaseModel):
    features_matrix: list[list[float]]


KERNEL_CODE = """
__global__ void dot_product(const float* matrix, const float* weights, float* out, int num_features, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0;
        for (int j = 0; j < num_features; j++) {
            sum += matrix[row * num_features + j] * weights[j];
        }
        out[row] = sum;
    }
}
"""


def gpu_inference(features_batch: np.ndarray, weights: np.ndarray, device_id: int, result_container: dict, key: str):
    device = cuda.Device(device_id)
    ctx = device.make_context()
    try:
        mod = SourceModule(KERNEL_CODE)
        dot_product = mod.get_function("dot_product")

        num_rows, num_features = features_batch.shape

        matrix_gpu = cuda.mem_alloc(features_batch.nbytes)
        cuda.memcpy_htod(matrix_gpu, features_batch)
        weights_gpu = cuda.mem_alloc(weights.nbytes)
        cuda.memcpy_htod(weights_gpu, weights)

        result_gpu = cuda.mem_alloc(num_rows * np.float32().nbytes)

        block_size = 256
        grid_size = (num_rows + block_size - 1) // block_size

        dot_product(
            matrix_gpu, weights_gpu, result_gpu,
            np.int32(num_features), np.int32(num_rows),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        result = np.empty(num_rows, dtype=np.float32)
        cuda.memcpy_dtoh(result, result_gpu)
        result_container[key] = result
    finally:
        ctx.pop()
        ctx.detach()


@app.post("/predict")
def predict(matrix_request: LinearRegressionRequest):
    data = np.array(matrix_request.features_matrix, dtype=np.float32)
    num_rows, num_features = data.shape

    # Эмитируем использование обученных весов линейной регрессии случайными весами.
    # Суть задачи это не меняет, просто в случае наличия обученных весов их нужно загрузить из файла, а не генерировать.
    weights = np.random.randn(num_features).astype(np.float32)

    num_devices = cuda.Device.count()

    results = {}

    # Если доступно более одного GPU, используем два потока
    if num_devices >= 2 and num_rows > 1:
        mid = num_rows // 2
        data_part1 = data[:mid]
        data_part2 = data[mid:]

        threads = []

        t1 = threading.Thread(target=gpu_inference, args=(data_part1, weights, 0, results, "part1"))
        t2 = threading.Thread(target=gpu_inference, args=(data_part2, weights, 1, results, "part2"))
        threads.append(t1)
        threads.append(t2)

        t1.start()
        t2.start()

        for t in threads:
            t.join()

        predictions = np.concatenate([results.get("part1", np.empty(0)), results.get("part2", np.empty(0))])
    else:
        # Если только один GPU или недостаточно строк для разделения, вычисляем всё на устройстве 0.
        gpu_inference(data, weights, 0, results, "result")
        predictions = results["result"]

    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1337)
