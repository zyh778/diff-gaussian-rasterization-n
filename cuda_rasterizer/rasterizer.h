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
 * @file rasterizer.h
 * @brief CUDA高斯光栅化器的主要接口定义
 * 
 * 该文件定义了高斯可微分光栅化的核心接口，包括前向渲染、反向传播和可见性标记功能
 * 这是整个CUDA光栅化系统的主要入口点
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	/**
	 * @brief CUDA高斯光栅化器类
	 * 
	 * 提供高斯可微分光栅化的核心功能，包括：
	 * - 前向渲染：将3D高斯转换为2D图像
	 * - 反向传播：计算梯度用于优化
	 * - 可见性标记：确定哪些高斯在当前视角下可见
	 */
	class Rasterizer
	{
	public:
		/**
		 * @brief 标记高斯在当前视角下的可见性
		 * 
		 * 通过视锥体裁剪确定哪些3D高斯在当前相机视角下可见
		 * 这是渲染流程的第一步，用于过滤不需要处理的高斯
		 * 
		 * @param P 高斯点的数量
		 * @param means3D 3D高斯中心点坐标数组
		 * @param viewmatrix 视图变换矩阵
		 * @param projmatrix 投影变换矩阵
		 * @param present 输出的可见性标记数组，true表示可见
		 */
		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		/**
		 * @brief 高斯光栅化的前向渲染过程
		 * 
		 * 执行完整的3D高斯到2D图像的渲染流程，包括：
		 * 1. 几何预处理（变换、投影、协方差计算）
		 * 2. 瓦片分箱和排序
		 * 3. Alpha混合渲染
		 * 
		 * @param geometryBuffer 几何状态缓冲区分配函数
		 * @param binningBuffer 分箱状态缓冲区分配函数
		 * @param imageBuffer 图像状态缓冲区分配函数
		 * @param P 高斯点的数量
		 * @param D 球谐函数的度数
		 * @param M 球谐函数的系数数量
		 * @param background 背景颜色（RGB）
		 * @param width 输出图像宽度
		 * @param height 输出图像高度
		 * @param means3D 3D高斯中心点坐标
		 * @param shs 球谐函数系数（用于颜色计算）
		 * @param colors_precomp 预计算的颜色（可选，替代球谐函数）
		 * @param opacities 高斯不透明度值
		 * @param scales 高斯缩放参数
		 * @param scale_modifier 全局缩放修饰符
		 * @param rotations 高斯旋转四元数
		 * @param cov3D_precomp 预计算的3D协方差矩阵（可选）
		 * @param viewmatrix 视图变换矩阵
		 * @param projmatrix 投影变换矩阵
		 * @param cam_pos 相机世界坐标位置
		 * @param tan_fovx X方向视场角的正切值
		 * @param tan_fovy Y方向视场角的正切值
		 * @param prefiltered 是否已预过滤高斯
		 * @param out_color 输出颜色缓冲区
		 * @param out_depth 输出深度缓冲区
		 * @param out_opacity 输出不透明度缓冲区
		 * @param radii 高斯投影半径（可选）
		 * @param is_used 高斯使用标记（可选）
		 * @param debug 是否启用调试模式
		 * @return int 实际渲染的高斯实例数量
		 */
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			float* out_opacity,
			int* radii = nullptr,
			int* is_used = nullptr,
			bool debug = false);

		/**
		 * @brief 高斯光栅化的反向传播过程
		 * 
		 * 计算前向渲染过程中所有参数的梯度，用于基于梯度的优化
		 * 从像素级损失梯度反向传播到所有高斯参数的梯度
		 * 
		 * @param P 高斯点的数量
		 * @param D 球谐函数的度数
		 * @param M 球谐函数的系数数量
		 * @param R 实际渲染的高斯实例数量
		 * @param background 背景颜色
		 * @param width 图像宽度
		 * @param height 图像高度
		 * @param means3D 3D高斯中心点坐标
		 * @param shs 球谐函数系数
		 * @param colors_precomp 预计算的颜色（可选）
		 * @param scales 高斯缩放参数
		 * @param scale_modifier 全局缩放修饰符
		 * @param rotations 高斯旋转四元数
		 * @param cov3D_precomp 预计算的3D协方差矩阵（可选）
		 * @param viewmatrix 视图变换矩阵
		 * @param projmatrix 投影变换矩阵
		 * @param projmatrix_raw 原始投影矩阵
		 * @param campos 相机世界坐标位置
		 * @param tan_fovx X方向视场角的正切值
		 * @param tan_fovy Y方向视场角的正切值
		 * @param radii 高斯投影半径
		 * @param geom_buffer 几何状态缓冲区（来自前向传播）
		 * @param binning_buffer 分箱状态缓冲区（来自前向传播）
		 * @param image_buffer 图像状态缓冲区（来自前向传播）
		 * @param dL_dpix 像素颜色损失梯度
		 * @param dL_dpix_depth 像素深度损失梯度
		 * @param dL_dmean2D 输出：2D中心点梯度
		 * @param dL_dconic 输出：圆锥矩阵梯度
		 * @param dL_dopacity 输出：不透明度梯度
		 * @param dL_dcolor 输出：颜色梯度
		 * @param dL_ddepths 输出：深度梯度
		 * @param dL_dmean3D 输出：3D中心点梯度
		 * @param dL_dcov3D 输出：3D协方差矩阵梯度
		 * @param dL_dsh 输出：球谐函数系数梯度
		 * @param dL_dscale 输出：缩放参数梯度
		 * @param dL_drot 输出：旋转参数梯度
		 * @param dL_dtau 输出：tau参数梯度
		 * @param debug 是否启用调试模式
		 */
		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
            const float* projmatrix_raw,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dpix_depth,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_ddepths,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dtau,
			bool debug);
	};
};

#endif