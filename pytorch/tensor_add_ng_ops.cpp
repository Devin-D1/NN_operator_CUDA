#include <torch/extension.h>
#include "tensor_add_ng.h"

void torch_launch_tensor_add_ng(torch::Tensor &c,
								const torch::Tensor &a,
								const torch::Tensor &b,
								int64_t n)
{
	tensor_add_ng((float *)c.data_ptr(),
				  (const float *)a.data_ptr(),
				  (const float *)b.data_ptr(),
				  n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("torch_launch_tensor_add_ng",
		  &torch_launch_tensor_add_ng,
		  "tensor_add_ng wrapper");
}
