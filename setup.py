#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
可微分高斯光栅化MCMC版本的安装配置文件

本文件使用setuptools配置Python包的编译和安装过程，
特别是配置CUDA扩展模块的编译参数和源文件。

主要功能：
- 定义包名称和结构
- 配置CUDA扩展模块的编译
- 指定源文件和编译参数
- 设置第三方库的包含路径
"""

from setuptools import setup  # Python包安装工具
from torch.utils.cpp_extension import CUDAExtension, BuildExtension  # PyTorch CUDA扩展工具
import os  # 操作系统接口模块

# 获取当前文件所在目录的绝对路径
os.path.dirname(os.path.abspath(__file__))

# 包安装配置
setup(
    # 包名称，用于pip安装时的标识
    name="diff_gaussian_rasterization_n",
    
    # 指定要包含的Python包目录
    packages=['diff_gaussian_rasterization_n'],
    
    # 定义CUDA扩展模块
    ext_modules=[
        CUDAExtension(
            # 扩展模块的完整名称，_C表示这是一个C++扩展
            name="diff_gaussian_rasterization_n._C",
            
            # 需要编译的源文件列表
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",  # 光栅化器主要实现
                "cuda_rasterizer/forward.cu",          # 前向传播CUDA核函数
                "cuda_rasterizer/backward.cu",         # 反向传播CUDA核函数
                "cuda_rasterizer/utils.cu",            # 工具函数CUDA实现
                "rasterize_points.cu",                 # 点光栅化主接口
                "ext.cpp"                              # Python绑定接口
            ],
            
            # NVCC编译器的额外参数
            # 添加GLM数学库的头文件路径到包含目录
            extra_compile_args={
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]
            }
        )
    ],
    
    # 自定义构建命令，使用PyTorch的BuildExtension来处理CUDA编译
    cmdclass={
        'build_ext': BuildExtension
    }
)
