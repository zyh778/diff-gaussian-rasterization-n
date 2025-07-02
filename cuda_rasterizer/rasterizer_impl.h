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
 * @file rasterizer_impl.h
 * @brief CUDA高斯光栅化器实现的内部数据结构和辅助函数
 * 
 * 该文件定义了光栅化过程中使用的内部状态结构：
 * - GeometryState: 几何处理阶段的状态数据
 * - ImageState: 图像渲染阶段的状态数据
 * - BinningState: 分箱排序阶段的状态数据
 * - 内存管理辅助函数
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

/**
 * @namespace CudaRasterizer
 * @brief CUDA高斯光栅化器的命名空间
 */
namespace CudaRasterizer
{
	/**
	 * @brief 从内存块中按对齐要求分配指定类型的数组
	 * 
	 * 该模板函数用于在连续内存块中按指定对齐要求分配数组空间
	 * 确保内存访问的效率和正确性
	 * 
	 * @tparam T 要分配的数据类型
	 * @param chunk 输入/输出：内存块指针，函数执行后会更新到下一个可用位置
	 * @param ptr 输出：分配的类型化指针
	 * @param count 要分配的元素数量
	 * @param alignment 内存对齐要求（字节）
	 */
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	/**
	 * @brief 几何处理阶段的状态数据结构
	 * 
	 * 存储高斯几何处理过程中需要的所有中间数据和结果
	 * 包括深度、坐标、协方差、颜色等信息
	 */
	struct GeometryState
	{
		size_t scan_size;              ///< 扫描操作所需的内存大小
		float* depths;                 ///< 高斯的深度值数组
		char* scanning_space;          ///< 扫描操作的工作空间
		bool* clamped;                 ///< 球谐函数是否被截断的标记数组
		int* internal_radii;           ///< 内部使用的半径数组
		float2* means2D;               ///< 高斯在2D屏幕空间的中心坐标
		float* cov3D;                  ///< 3D协方差矩阵数组
		float4* conic_opacity;         ///< 2D协方差逆矩阵和不透明度
		float* rgb;                    ///< RGB颜色数组
		float3* normals;               ///< 法线向量数组
		uint32_t* point_offsets;       ///< 点的偏移量数组
		uint32_t* tiles_touched;       ///< 每个高斯覆盖的瓦片数量

		/**
		 * @brief 从内存块创建GeometryState对象
		 * @param chunk 内存块指针
		 * @param P 高斯点的数量
		 * @return GeometryState 创建的状态对象
		 */
		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	/**
	 * @brief 图像渲染阶段的状态数据结构
	 * 
	 * 存储图像渲染过程中每个像素的状态信息
	 * 用于alpha混合和最终颜色计算
	 */
	struct ImageState
	{
		uint2* ranges;                 ///< 每个瓦片的高斯索引范围
		uint32_t* n_contrib;           ///< 每个像素的贡献高斯数量
		float* accum_alpha;            ///< 累积的alpha值

		/**
		 * @brief 从内存块创建ImageState对象
		 * @param chunk 内存块指针
		 * @param N 像素或瓦片的数量
		 * @return ImageState 创建的状态对象
		 */
		static ImageState fromChunk(char*& chunk, size_t N);
	};

	/**
	 * @brief 分箱排序阶段的状态数据结构
	 * 
	 * 存储高斯按瓦片分组和深度排序过程中需要的数据
	 * 用于确保正确的alpha混合顺序
	 */
	struct BinningState
	{
		size_t sorting_size;           ///< 排序操作所需的内存大小
		uint64_t* point_list_keys_unsorted; ///< 未排序的键值对数组
		uint64_t* point_list_keys;     ///< 排序后的键值对数组
		uint32_t* point_list_unsorted; ///< 未排序的高斯索引列表
		uint32_t* point_list;          ///< 排序后的高斯索引列表
		char* list_sorting_space;      ///< 排序操作的工作空间

		/**
		 * @brief 从内存块创建BinningState对象
		 * @param chunk 内存块指针
		 * @param P 高斯点的数量
		 * @return BinningState 创建的状态对象
		 */
		static BinningState fromChunk(char*& chunk, size_t P);
	};

	/**
	 * @brief 计算指定状态类型所需的内存大小
	 * 
	 * 该模板函数计算创建指定类型状态对象所需的总内存大小
	 * 包括额外的128字节缓冲区以确保内存安全
	 * 
	 * @tparam T 状态类型（GeometryState、ImageState或BinningState）
	 * @param P 高斯点或像素的数量
	 * @return size_t 所需的内存大小（字节）
	 */
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};