import time
import torch
import argparse
import numpy as np

print("Torch version: " + torch.__version__)

n = 1024 * 1024
a = torch.rand(n, device='cuda:0')
b = torch.rand(n, device='cuda:0')
cuda_c = torch.rand(n, device='cuda:0')
testn = 10

def record_time(fun):
    times = list()
    res = None

    # GPU Warm up
    for _ in range(0, testn):
        res = fun()

    for _ in range(0, testn):
        torch.cuda.synchronize(device="cuda:0")
        start = time.time()
        fun()
        torch.cuda.synchronize(device="cuda:0")
        end = time.time()
        times.append((end - start) * 1e6)
    return times, res

def run_torch():
    c = a + b
    return c.contiguous()

def run_cuda():
    cuda_func(cuda_c, a, b, n)
    return cuda_c

def pre_load():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=str, choices=['jit', 'setup', 'cmake'], default='jit')
    args = parser.parse_args()
    res = None

    if args.compiler == 'jit':
        from torch.utils.cpp_extension import load
        module = load(name='tiny_operator',
                      extra_include_paths=["include"],
                      sources=["pytorch/tiny_operator.cpp", "kernel/tensor_add_ng.cu"],
                      verbose=True)
        res = module.torch_launch_tensor_add_ng
    elif args.compiler == 'setup':
        import tensor_add_ng
        res = tensor_add_ng.torch_launch_tensor_add_ng
    elif args.compiler == 'cmake':
        torch.ops.load_library("build/tiny_operator.so")
        res = torch.ops.tiny_operator.torch_launch_tensor_add_ng
    else:
        raise Exception("Type of cuda compiler must be one of jit/setup/cmake.")

    return res

if __name__ == "__main__":
    cuda_func = pre_load()

    print("Run torch...")
    torch_time, torch_res = record_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    print("Run CUDA operator...")
    cuda_time, cuda_res = record_time(run_cuda)
    print("CUDA time:  {:.3f}us".format(np.mean(cuda_time)))

    assert(torch_res.equal(cuda_res))
