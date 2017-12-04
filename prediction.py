import mvnc.mvncapi as mvnc
import numpy as np


def predict(input):
    devices = mvnc.EnumerateDevices()
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    with open('./model/graph', 'rb') as f:
        blob = f.read()
    graph = device.AllocateGraph(blob)

    for i in range(4):
        graph.LoadTensor(input[i], 'user object')
        output, userobj = graph.GetResult()
        print(np.argmax(output))

    graph.DeallocateGraph()
    device.CloseDevice()

    return output


if __name__ == "__main__":
    input = np.load("test_data.npy")
    predict(input.astype(np.float16))
