import time
import torch
import numpy as np
from torch.utils.cpp_extension import load

print("Torch version: " + torch.__version__)

n = 1024 * 1024
a = torch.rand(n, device='cuda:0')
b = torch.rand(n, device='cuda:0')
cuda_c = torch.rand(n, device='cuda:0')
testn = 10

def record_time(fun):
    times = list()

    # GPU Warm up
    for _ in range(0, testn):
        fun()

    for _ in range(0, testn):
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

def run_cuda():
    cuda_module.torch_launch_tensor_add_ng(cuda_c, a, b, n)

def pre_load():
    module = load(name='torch_operator',
                  extra_include_paths=["include"],
                  sources=["pytorch/tensor_add_ng_ops.cpp", "kernel/tensor_add_ng.cu"],
                  verbose=True)
    return module

if __name__ == "__main__":
    cuda_module = pre_load()

    print("Run torch...")
    torch_time = record_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    print("Run CUDA operator...")
    cuda_time = record_time(run_cuda)
    print("CUDA time:  {:.3f}us".format(np.mean(cuda_time)))
