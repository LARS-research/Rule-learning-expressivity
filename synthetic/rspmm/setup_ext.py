from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torch_ext',
      ext_modules=[cpp_extension.CppExtension(
            name='torch_ext', 
            sources=['torch_ext.cpp'],
            extra_cflags=["-Ofast", "-g", "-DAT_PARALLEL_NATIVE", "-DCUDA_OP"],
            extra_cuda_cflags=["-O3", ""],
            extra_ldflags=["-ltorch"]
            )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})