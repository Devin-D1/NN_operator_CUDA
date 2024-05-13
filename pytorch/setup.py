from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = "tiny_operator",
    include_dirs = ["include"],
    ext_modules = [
        CUDAExtension(
            "tensor_add_ng",
            ["pytorch/tensor_add_ng_ops.cpp", "kernel/tensor_add_ng.cu"],
        )
    ],
    cmdclass = {
        "build_ext": BuildExtension
    }
)
