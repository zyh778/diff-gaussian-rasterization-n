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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/utils.h"
#include <fstream>
#include <string>
#include <functional>

// 定义一个用于调整张量大小的函数对象
// 输入参数t是一个torch::Tensor引用
// 返回一个函数对象,该函数接受size_t类型的参数N,用于指定新的张量大小
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// 高斯点云的前向渲染函数
// 输入包括:背景图、3D点位置、颜色、不透明度、尺度、旋转等参数
// 输出渲染结果、深度图和其他缓冲信息
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,    // 背景图像
	const torch::Tensor& means3D,       // 3D点云位置
    const torch::Tensor& colors,        // 点云颜色
    const torch::Tensor& opacity,       // 不透明度
	const torch::Tensor& scales,        // 尺度
	const torch::Tensor& rotations,     // 旋转
	const float scale_modifier,         // 尺度修改器
	const torch::Tensor& cov3D_precomp, // 预计算的3D协方差
	const torch::Tensor& viewmatrix,    // 视图矩阵
	const torch::Tensor& projmatrix,    // 投影矩阵
    const torch::Tensor& projmatrix_raw,// 原始投影矩阵
	const float tan_fovx,              // 水平视场角的正切
	const float tan_fovy,              // 垂直视场角的正切
    const int image_height,            // 图像高度
    const int image_width,             // 图像宽度
	const torch::Tensor& sh,           // 球谐系数
	const int degree,                  // 球谐度数
	const torch::Tensor& campos,       // 相机位置
	const bool prefiltered,            // 是否预过滤
	const bool debug)                  // 是否开启调试
{
  // 检查输入数据维度是否正确
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  // 获取基本参数
  const int P = means3D.size(0);  // 点云数量
  const int H = image_height;     // 图像高度
  const int W = image_width;      // 图像宽度

  // 设置张量数据类型选项
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  // 初始化输出张量
  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);    // 输出颜色图
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));// 点云半径
  torch::Tensor is_used = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));// 使用标志
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);               // 输出深度图
  torch::Tensor out_normal = torch::full({3, H, W}, 0.0, float_opts);              // 输出法线图
  torch::Tensor out_opacity = torch::full({1, H, W}, 0.0, float_opts);             // 输出不透明度图
  // 创建CUDA设备上的缓冲区
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));      // 几何缓冲区
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));   // 分箱缓冲区
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));       // 图像缓冲区
  
  // 创建缓冲区调整函数
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  // 执行渲染
  int rendered = 0;
  if(P != 0)
  {
	  // 获取球谐系数维度
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  // 调用CUDA光栅化器进行前向渲染
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_normal.contiguous().data<float>(),
		out_opacity.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		is_used.contiguous().data<int>(),
		debug);
  }
  
  // 返回渲染结果
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_depth, out_normal,out_opacity, is_used);
}

// 高斯点云的反向传播函数
// 计算各个参数对应的梯度
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,    // 背景图像
	const torch::Tensor& means3D,       // 3D点位置
	const torch::Tensor& radii,         // 点云半径
    const torch::Tensor& colors,        // 颜色
	const torch::Tensor& scales,        // 尺度
	const torch::Tensor& rotations,     // 旋转
	const float scale_modifier,         // 尺度修改器
	const torch::Tensor& cov3D_precomp, // 预计算的3D协方差
	const torch::Tensor& viewmatrix,    // 视图矩阵
    const torch::Tensor& projmatrix,    // 投影矩阵
    const torch::Tensor& projmatrix_raw,// 原始投影矩阵
	const float tan_fovx,              // 水平视场角的正切
	const float tan_fovy,              // 垂直视场角的正切
    const torch::Tensor& dL_dout_color, // 颜色的梯度
	const torch::Tensor& dL_dout_depths,// 深度的梯度
	const torch::Tensor& dL_dout_normal,// 法线的梯度
	const torch::Tensor& sh,           // 球谐系数
	const int degree,                  // 球谐度数
	const torch::Tensor& campos,       // 相机位置
	const torch::Tensor& geomBuffer,   // 几何缓冲区
	const int R,                       // 渲染的点数
	const torch::Tensor& binningBuffer,// 分箱缓冲区
	const torch::Tensor& imageBuffer,  // 图像缓冲区
	const bool debug)                  // 是否开启调试
{
  // 获取基本参数
  const int P = means3D.size(0);      // 点云数量
  const int H = dL_dout_color.size(1);// 图像高度
  const int W = dL_dout_color.size(2);// 图像宽度
  
  // 获取球谐系数维度
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  // 初始化梯度张量
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());        // 3D位置梯度
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());        // 2D位置梯度
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());// 颜色梯度
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());         // 深度梯度
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());       // 圆锥梯度
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());        // 不透明度梯度
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());          // 3D协方差梯度
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());          // 球谐系数梯度
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());         // 尺度梯度
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());      // 旋转梯度
  torch::Tensor dL_dtau = torch::zeros({P,6}, means3D.options());             // tau梯度
  torch::Tensor dL_dnormals = torch::zeros({P, 3}, means3D.options());        // 法线梯度
  
  // 执行反向传播
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
      projmatrix_raw.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depths.contiguous().data<float>(),
	  dL_dout_normal.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dnormals.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
      dL_dtau.contiguous().data<float>(),
	  debug);
  }

  // 返回计算的梯度
  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dtau, dL_dnormals);
}

// 标记可见点云的函数
// 根据视图矩阵和投影矩阵判断每个3D点是否可见
torch::Tensor markVisible(
		torch::Tensor& means3D,      // 3D点位置
		torch::Tensor& viewmatrix,   // 视图矩阵
		torch::Tensor& projmatrix)   // 投影矩阵
{ 
  const int P = means3D.size(0);    // 点云数量
  
  // 初始化可见性标记张量
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  // 执行可见性判断
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

// 计算重定位的函数
// 根据旧的不透明度和尺度计算新的不透明度和尺度
std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,    // 旧的不透明度
	torch::Tensor& scale_old,      // 旧的尺度
	torch::Tensor& N,              // 点数
	torch::Tensor& binoms,         // 二项式系数
	const int n_max)              // 最大点数
{
	const int P = opacity_old.size(0);  // 点云数量
  
	// 初始化输出张量
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	// 执行重定位计算
	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);
}