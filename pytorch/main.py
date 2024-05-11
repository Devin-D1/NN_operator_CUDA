import time
import torch
import numpy as np

print("Torch version: " + torch.__version__)

n = 1024 * 1024
a = torch.rand(n, device='cuda:0')
b = torch.rand(n, device='cuda:0')
cuda_c = torch.rand(n, device='cuda:0')
n = 10

def record_time(fun):
    times = list()

    # GPU Warm up
    for _ in range(0, n):
        fun()

    for _ in range(0, n):
        torch.cuda.synchronize(device="cuda:0")
        start = time.time()
        fun()
        torch.cuda.synchronize(device="cuda:0")
        end = time.time()
        times.append((end - start) * 1e6)
    return times


def run_torch():
    c = a + b
    return c.contiguous()

if __name__ == "__main__":
    print("Run torch...")
    torch_time = record_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))
