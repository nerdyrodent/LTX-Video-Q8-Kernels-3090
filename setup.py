import subprocess
import os
import torch
import platform
from packaging.version import parse, Version
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


# package name managed by pip, which can be remove by `pip uninstall tiny_pkg`
PACKAGE_NAME = "q8_kernels"
system_name = platform.system()

ext_modules = []
generator_flag = []
cc_flag = []
cc_flag.append("--gpu-architecture=native")

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

def get_device_arch():
    major, minor = torch.cuda.get_device_capability(0)
    if major == 8 and (minor >= 0 and minor < 9):
        return "ampere"
    if major == 8 and minor == 9:
        return "ada"
    if major == 9 and minor == 0:
        return "hopper"
    if major == 12:
        return "blackwell"
    raise NotImplementedError("Not supported gpu!")
    
this_dir = Path(__file__).parent
device_arch = get_device_arch()
should_compile_fp8_fast_acc = device_arch in ["ada", "blackwell"]
if should_compile_fp8_fast_acc:
    subprocess.run(["git", "submodule", "update", "--init", "third_party/cutlass"], check=True)

ext_modules.append(
    CUDAExtension(
        # package name for import
        name="q8_kernels_cuda.ops._C",
        sources=[
            "csrc/fast_hadamard/fast_hadamard_transform.cpp",
            "csrc/ops/ops_api.cpp",
            "csrc/fast_hadamard/fast_hadamard_transform_cuda.cu",
            "csrc/fast_hadamard/fused_hadamard_transform_cuda.cu",
            "csrc/fast_hadamard/rms_norm_rope_cuda.cu",
            "csrc/fast_hadamard/dequant_fast_hadamard_transform_cuda.cu"
        ],
          extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-lineinfo",
                    "--ptxas-options=-v",
                    "--ptxas-options=-O2",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",

                ]
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            this_dir / "csrc" / "fast_hadamard",
        ],
    )
)

if should_compile_fp8_fast_acc:
    ext_modules.append(
        CUDAExtension(
            name="q8_kernels_cuda.gemm._C",
            sources=[
                "csrc/gemm/fp8_gemm.cpp",
                "csrc/gemm/fp8_gemm_cuda.cu",
                "csrc/gemm/fp8_gemm_bias.cu",
                
            ],
            extra_compile_args={
                # add c compile flags
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-lineinfo",
                        "--ptxas-options=-v",
                        "--ptxas-options=-O2",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    ]
                    + generator_flag
                    + cc_flag,
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "gemm",
                Path(this_dir) / "third_party/cutlass/include",
                Path(this_dir) / "third_party/cutlass/tools/utils/include" ,
                Path(this_dir) / "third_party/cutlass/examples/common" ,
            ],
        )
    )

    ext_modules.append(
        CUDAExtension(
            name="q8_kernels_cuda.flash_attention._C",
            sources=[
                "csrc/flash_attention/flash_attention.cpp",
                "csrc/flash_attention/flash_attention_cuda.cu",
                
            ],
            extra_compile_args={
                # add c compile flags
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-lineinfo",
                        "--ptxas-options=-v",
                        "--ptxas-options=-O2",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    ]
                    + generator_flag
                    + cc_flag,
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attention",
                Path(this_dir) / "third_party/cutlass/include",
                Path(this_dir) / "third_party/cutlass/tools/utils/include" ,
                Path(this_dir) / "third_party/cutlass/examples/common" ,
            ],
        )
    )

setup(
    name=PACKAGE_NAME,
    version="0.0.4",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="8bit kernels",
    ext_modules=ext_modules,
    cmdclass={ "build_ext": BuildExtension, "bdist_wheel": _bdist_wheel},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    author="KONAKONA666/Aibek Bekbayev (2025 Lightricks Ltd.)",
)