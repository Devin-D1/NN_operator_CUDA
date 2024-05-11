__global__ void tensor_add_kernel(float* c, const float* a, const float* b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	c[tid] = a[tid] + b[tid];
}

void tensor_add_ng(float* c, const float* a, const float* b, int n)
{
	dim3 grid((n + 1023) / 1024);
	dim3 block(1024);
	tensor_add_kernel<<<grid, block>>>(c, a, b);
}
