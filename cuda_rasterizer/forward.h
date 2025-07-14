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
 * @file forward.h
 * @brief CUDA高斯光栅化器前向传播函数声明
 * 
 * 该文件定义了高斯光栅化前向传播过程中的主要函数：
 * - preprocess: 高斯预处理，计算投影坐标、协方差等
 * - render: 主要光栅化渲染过程
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

/**
 * @namespace FORWARD
 * @brief 前向传播相关函数的命名空间
 */
namespace FORWARD
{
	/**
	 * @brief 高斯光栅化预处理函数
	 * 
	 * 在光栅化之前对每个高斯进行初始处理，包括：
	 * - 坐标变换（世界空间到屏幕空间）
	 * - 协方差矩阵计算
	 * - 颜色计算（球谐函数或预计算颜色）
	 * - 法线计算（预计算法线或默认法线）
	 * - 视锥体裁剪
	 * - 瓦片覆盖计算
	 * 
	 * @param P 高斯点的数量
	 * @param D 球谐函数的度数
	 * @param M 球谐函数系数的数量
	 * @param orig_points 原始3D点坐标
	 * @param scales 高斯的缩放参数
	 * @param scale_modifier 全局缩放修饰符
	 * @param rotations 高斯的旋转四元数
	 * @param opacities 高斯的不透明度
	 * @param shs 球谐函数系数
	 * @param clamped 输出：球谐函数是否被截断
	 * @param cov3D_precomp 预计算的3D协方差矩阵（可选）
	 * @param colors_precomp 预计算的颜色（可选）
	 * @param normals_precomp 预计算的法线（可选）
	 * @param viewmatrix 视图变换矩阵
	 * @param projmatrix 投影变换矩阵
	 * @param cam_pos 相机位置
	 * @param W 图像宽度
	 * @param H 图像高度
	 * @param focal_x X方向焦距
	 * @param focal_y Y方向焦距
	 * @param tan_fovx X方向视场角的正切值
	 * @param tan_fovy Y方向视场角的正切值
	 * @param radii 输出：高斯在屏幕空间的半径
	 * @param points_xy_image 输出：高斯在图像空间的2D坐标
	 * @param depths 输出：高斯的深度值
	 * @param cov3Ds 输出：3D协方差矩阵
	 * @param colors 输出：计算得到的颜色
	 * @param normals 输出：计算得到的法线向量
	 * @param conic_opacity 输出：2D协方差的逆矩阵和不透明度
	 * @param grid 瓦片网格维度
	 * @param tiles_touched 输出：每个高斯覆盖的瓦片数量
	 * @param prefiltered 是否已经预过滤
	 */
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* normals_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float* normals,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	/**
	 * @brief 主要光栅化渲染函数
	 * 
	 * 执行实际的光栅化渲染过程，将高斯混合到最终图像中
	 * 使用alpha混合技术，按深度顺序处理每个像素的高斯贡献
	 * 
	 * @param grid CUDA网格维度
	 * @param block CUDA线程块维度
	 * @param ranges 每个瓦片中高斯的索引范围
	 * @param point_list 排序后的高斯索引列表
	 * @param W 图像宽度
	 * @param H 图像高度
	 * @param points_xy_image 高斯在图像空间的2D坐标
	 * @param features 高斯的特征（颜色）
	 * @param conic_opacity 2D协方差的逆矩阵和不透明度
	 * @param final_T 输出：最终的透射率
	 * @param n_contrib 输出：每个像素的贡献高斯数量
	 * @param bg_color 背景颜色
	 * @param out_color 输出：渲染的颜色图像
	 * @param depth 高斯的深度值
	 * @param out_depth 输出：深度图像
	 * @param out_opacity 输出：不透明度图像
	 * @param is_used 输出：高斯是否被使用的标记
	 */
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float* normals,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		const float* depth,
		float* out_depth,
		float* out_normals,
		float* out_opacity,
		int* is_used);
}


#endif