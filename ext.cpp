/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

/**
 * PyTorch C++扩展模块入口文件
 * 
 * 本文件定义了Python与C++/CUDA代码之间的接口绑定，
 * 使用pybind11库将C++函数暴露给Python环境。
 * 
 * 主要功能：
 * - 绑定3D高斯溅射的前向渲染函数
 * - 绑定反向传播梯度计算函数
 * - 绑定可见性标记函数
 * - 绑定重定位计算函数
 */

#include <torch/extension.h>  // PyTorch C++扩展头文件
#include "rasterize_points.h"  // 高斯点光栅化函数声明

/**
 * PyBind11模块定义
 * 
 * 使用PYBIND11_MODULE宏定义Python模块，将C++函数绑定到Python接口
 * TORCH_EXTENSION_NAME是PyTorch自动生成的模块名称
 * 
 * @param m 模块对象，用于注册函数
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 绑定高斯点云前向渲染函数
  // 将3D高斯椭球体渲染成2D图像
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  
  // 绑定高斯点云反向传播函数
  // 计算渲染过程中各参数的梯度，用于训练优化
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  
  // 绑定可见性标记函数
  // 快速判断哪些高斯点在当前视角下可见，用于剔除优化
  m.def("mark_visible", &markVisible);
  
  // 绑定重定位计算函数
  // 用于MCMC采样过程中的高斯点重定位计算
  m.def("compute_relocation", &ComputeRelocationCUDA);
}