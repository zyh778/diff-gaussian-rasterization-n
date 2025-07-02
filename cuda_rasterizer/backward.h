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
 * @file backward.h
 * @brief CUDA高斯光栅化器反向传播函数声明
 * 
 * 该文件定义了高斯光栅化反向传播过程中的主要函数：
 * - render: 渲染阶段的反向传播，计算像素梯度到高斯参数的梯度
 * - preprocess: 预处理阶段的反向传播，计算高斯参数的梯度
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

/**
 * @namespace BACKWARD
 * @brief 反向传播相关函数的命名空间
 */
namespace BACKWARD
{
	/**
	 * @brief 渲染阶段的反向传播函数
	 * 
	 * 从像素损失梯度反向传播到高斯的2D参数梯度
	 * 计算每个高斯对最终渲染结果的贡献梯度
	 * 
	 * @param grid CUDA网格维度
	 * @param block CUDA线程块维度
	 * @param ranges 每个瓦片中高斯的索引范围
	 * @param point_list 排序后的高斯索引列表
	 * @param W 图像宽度
	 * @param H 图像高度
	 * @param bg_color 背景颜色
	 * @param means2D 高斯在图像空间的2D坐标
	 * @param conic_opacity 2D协方差的逆矩阵和不透明度
	 * @param colors 高斯的颜色
	 * @param depths 高斯的深度值
	 * @param final_Ts 最终的透射率
	 * @param n_contrib 每个像素的贡献高斯数量
	 * @param dL_dpixels 像素颜色的损失梯度
	 * @param dL_dpixels_depth 像素深度的损失梯度
	 * @param dL_dpixels_normal 像素法向量的损失梯度
	 * @param dL_dmean2D 输出：2D坐标的梯度
	 * @param dL_dconic2D 输出：2D协方差逆矩阵的梯度
	 * @param dL_dopacity 输出：不透明度的梯度
	 * @param dL_dcolors 输出：颜色的梯度
	 * @param dL_ddepths 输出：深度的梯度
	 * @param dL_dnormals 输出：法向量的梯度
	 */
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float3* normals,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dpixels_depth,
		const float* dL_dpixels_normal,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_ddepths,
		float3* dL_dnormals);

	/**
	 * @brief 预处理阶段的反向传播函数
	 * 
	 * 从2D参数梯度反向传播到原始高斯参数梯度
	 * 计算3D位置、缩放、旋转、球谐函数系数等参数的梯度
	 * 
	 * @param P 高斯点的数量
	 * @param D 球谐函数的度数
	 * @param M 球谐函数系数的数量
	 * @param means 高斯的3D中心位置
	 * @param radii 高斯在屏幕空间的半径
	 * @param shs 球谐函数系数
	 * @param clamped 球谐函数是否被截断
	 * @param scales 高斯的缩放参数
	 * @param rotations 高斯的旋转四元数
	 * @param scale_modifier 全局缩放修饰符
	 * @param cov3Ds 3D协方差矩阵
	 * @param view 视图变换矩阵
	 * @param proj 投影变换矩阵
	 * @param proj_raw 原始投影矩阵
	 * @param focal_x X方向焦距
	 * @param focal_y Y方向焦距
	 * @param tan_fovx X方向视场角的正切值
	 * @param tan_fovy Y方向视场角的正切值
	 * @param campos 相机位置
	 * @param dL_dmean2D 2D坐标的梯度
	 * @param dL_dconics 2D协方差逆矩阵的梯度
	 * @param dL_dmeans 输出：3D位置的梯度
	 * @param dL_dcolor 输出：颜色的梯度
	 * @param dL_ddepth 输出：深度的梯度
	 * @param dL_dcov3D 输出：3D协方差的梯度
	 * @param dL_dsh 输出：球谐函数系数的梯度
	 * @param dL_dscale 输出：缩放参数的梯度
	 * @param dL_drot 输出：旋转四元数的梯度
	 * @param dL_dtau 输出：tau参数的梯度
	 */
	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float* proj_raw,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_ddepth,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		float* dL_dtau);
}

#endif