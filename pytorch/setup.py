from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name ="tiny_operator",
    version ="0.0.1",
    include_dirs = ["include"],
    ext_modules = [
        CUDAExtension(
            "tensor_add_ng",
            ["pytorch/tiny_operator.cpp", "kernel/tensor_add_ng.cu"],
        )
    ],
    cmdclass = {
        "build_ext": BuildExtension
    }
)
