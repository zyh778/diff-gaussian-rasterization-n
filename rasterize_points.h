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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

// 前向渲染高斯点云的CUDA实现
// 输入参数:
// - background: 背景图像
// - means3D: 3D空间中高斯点的中心位置
// - colors: 高斯点的颜色
// - opacity: 高斯点的不透明度
// - scales: 高斯点的缩放系数
// - rotations: 高斯点的旋转参数
// - scale_modifier: 全局缩放修正系数
// - cov3D_precomp: 预计算的3D协方差矩阵
// - viewmatrix: 相机视图矩阵
// - projmatrix: 投影矩阵
// - projmatrix_raw: 原始投影矩阵
// - tan_fovx/y: 视场角的正切值
// - image_height/width: 输出图像尺寸
// - sh: 球谐函数系数
// - degree: 球谐函数的度数
// - campos: 相机位置
// - prefiltered: 是否预过滤
// - debug: 是否开启调试模式
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
    const torch::Tensor& projmatrix_raw,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

// 反向传播计算梯度的CUDA实现
// 除了与前向传播相同的参数外，还包括:
// - radii: 高斯点的半径
// - dL_dout_color: 输出颜色关于损失的梯度
// - dL_dout_depth: 输出深度关于损失的梯度
// - dL_dout_normal: 输出法线关于损失的梯度
// - geomBuffer: 几何缓冲区
// - R: 光栅化范围参数
// - binningBuffer: 空间划分缓冲区
// - imageBuffer: 图像缓冲区
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& projmatrix_raw,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_normal,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
// 标记视野内可见的高斯点
// 输入:
// - means3D: 3D点的位置
// - viewmatrix: 视图矩阵
// - projmatrix: 投影矩阵
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

// 计算高斯点的重定位参数
// 输入:
// - opacity_old: 原始不透明度
// - scale_old: 原始缩放
// - N: 采样数量
// - binoms: 二项式系数
// - n_max: 最大迭代次数
std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
		torch::Tensor& opacity_old,
		torch::Tensor& scale_old,
		torch::Tensor& N,
		torch::Tensor& binoms,
		const int n_max);