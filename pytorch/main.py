import torch

print("Torch version: " + torch.__version__)

n = 1024 * 1024
a = torch.rand(n, device='cuda:0')
b = torch.rand(n, device='cuda:0')
cuda_c = torch.rand(n, device='cuda:0')

def run_torch():
    c = a + b
    return c.contiguous()

if __name__ == "__main__":
    out = run_torch()
    print("First 10: " + str(out[10:]))
    print("Complete!")
