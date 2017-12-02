import mvnc.mvncapi as mvnc
import numpy as np

def predict(input):
    devices = mvnc.EnumerateDevices()
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    with open('graph', 'rb') as f:
        blob = f.read()
    graph = device.AllocateGraph(blob)

    graph.LoadTensor(input.astype(np.float16), 'user object')
    output, userobj = graph.GetResult()

    graph.DeallocateGraph()
    device.CloseDevice()

    return output

if __name__=="__main__":
    input = np.load("test_data.npy")
    print(np.argmax(predict(input)))