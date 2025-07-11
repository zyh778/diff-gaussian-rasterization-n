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
 * @file rasterizer_impl.cu
 * @brief 高斯光栅化器的CUDA实现
 * 
 * 本文件包含了3D高斯溅射光栅化的核心CUDA实现，包括：
 * - 视锥体裁剪和可见性检测
 * - 高斯到瓦片的映射和排序
 * - 前向和反向渲染过程
 * - 内存管理和状态管理
 * 
 * 主要功能：
 * - checkFrustum: 检查高斯是否在视锥体内
 * - duplicateWithKeys: 为高斯-瓦片重叠生成键值对
 * - identifyTileRanges: 识别瓦片范围
 * - forward/backward: 前向和反向渲染
 */

#include "rasterizer_impl.h"
#include <iostream>                          // 标准输入输出流
#include <fstream>                           // 文件流操作
#include <algorithm>                         // 算法库
#include <numeric>                           // 数值算法
#include <cuda.h>                            // CUDA运行时API
#include "cuda_runtime.h"                    // CUDA运行时头文件
#include "device_launch_parameters.h"        // CUDA设备启动参数
#include <cub/cub.cuh>                       // CUDA CUB库（CUDA UnBound）
#include <cub/device/device_radix_sort.cuh>  // CUB基数排序
#define GLM_FORCE_CUDA                       // 强制GLM使用CUDA
#include <glm/glm.hpp>                       // GLM数学库

#include <cooperative_groups.h>              // CUDA协作组
#include <cooperative_groups/reduce.h>       // 协作组归约操作
namespace cg = cooperative_groups;           // 协作组命名空间别名

#include "auxiliary.h"                       // 辅助函数
#include "forward.h"                         // 前向渲染
#include "backward.h"                        // 反向渲染

/**
 * @brief 在CPU上查找最高有效位(MSB)的下一个更高位
 * 
 * 这是一个辅助函数，用于计算给定数字的最高有效位位置
 * 主要用于内存对齐和位操作优化
 * 
 * @param n 输入的32位无符号整数
 * @return uint32_t 最高有效位的位置
 */
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;  // 初始化为32位
	uint32_t step = msb;           // 步长初始化
	while (step > 1)
	{
		step /= 2;                 // 二分查找步长
		if (n >> msb)              // 检查当前位置是否有位
			msb += step;           // 向上调整
		else
			msb -= step;           // 向下调整
	}
	if (n >> msb)                  // 最终检查
		msb++;
	return msb;
}

/**
 * @brief CUDA核函数：检查高斯是否在视锥体内
 * 
 * 这是一个包装方法，调用辅助的粗略视锥体包含测试
 * 标记所有通过测试的高斯点
 * 
 * @param P 高斯点的总数
 * @param orig_points 原始3D点坐标数组
 * @param viewmatrix 视图变换矩阵
 * @param projmatrix 投影变换矩阵
 * @param present 输出布尔数组，标记每个点是否可见
 */
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();  // 获取当前线程在网格中的全局索引
	if (idx >= P)                              // 边界检查
		return;

	float3 p_view;                             // 视图空间中的点坐标
	// 调用视锥体测试函数，检查点是否在视锥体内
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

/**
 * @brief CUDA核函数：为所有高斯/瓦片重叠生成键值对
 * 
 * 为每个高斯生成与其重叠的所有瓦片的键值对（1:N映射）
 * 这是光栅化过程中的关键步骤，用于后续的排序和渲染
 * 
 * @param P 高斯点的总数
 * @param points_xy 2D投影点坐标数组
 * @param depths 深度值数组
 * @param offsets 每个高斯在输出缓冲区中的偏移量
 * @param gaussian_keys_unsorted 未排序的高斯键数组（输出）
 * @param gaussian_values_unsorted 未排序的高斯值数组（输出）
 * @param radii 高斯半径数组
 * @param grid 瓦片网格维度
 */
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();  // 获取当前线程索引
	if (idx >= P)                              // 边界检查
		return;

	// 对不可见的高斯不生成键值对
	if (radii[idx] > 0)
	{
		// 找到该高斯在缓冲区中写入键值的偏移量
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;  // 边界矩形的最小和最大坐标

		// 计算高斯投影的边界矩形
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// 为边界矩形重叠的每个瓦片生成一个键值对
		// 键的格式是 | 瓦片ID | 深度值 |
		// 值是高斯的ID。使用这个键对值进行排序可以得到高斯ID列表，
		// 首先按瓦片排序，然后按深度排序
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// 计算瓦片ID并构造64位键
				uint64_t key = y * grid.x + x;           // 瓦片ID
				key <<= 32;                              // 左移32位为深度留出空间
				key |= *((uint32_t*)&depths[idx]);       // 添加深度值
				gaussian_keys_unsorted[off] = key;       // 存储键
				gaussian_values_unsorted[off] = idx;     // 存储高斯ID
				off++;                                   // 移动到下一个位置
			}
		}
	}
}

/**
 * @brief CUDA核函数：识别瓦片范围
 * 
 * 检查键值以确定是否位于完整排序列表中某个瓦片范围的开始/结束位置
 * 如果是，则写入该瓦片的开始/结束位置
 * 对每个实例化（重复的）高斯ID运行一次
 * 
 * @param L 排序后的键值对总数
 * @param point_list_keys 排序后的键数组
 * @param ranges 输出的瓦片范围数组，每个元素包含开始和结束索引
 */
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();  // 获取当前线程索引
	if (idx >= L)                              // 边界检查
		return;

	// 从键中读取瓦片ID，如果在边界处则更新瓦片范围的开始/结束
	uint64_t key = point_list_keys[idx];       // 当前键
	uint32_t currtile = key >> 32;             // 提取瓦片ID（高32位）
	if (idx == 0)                              // 第一个元素
		ranges[currtile].x = 0;                // 设置当前瓦片的开始位置
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;  // 前一个瓦片ID
		if (currtile != prevtile)                            // 瓦片ID发生变化
		{
			ranges[prevtile].y = idx;                       // 设置前一个瓦片的结束位置
			ranges[currtile].x = idx;                       // 设置当前瓦片的开始位置
		}
	}
	if (idx == L - 1)                          // 最后一个元素
		ranges[currtile].y = L;                // 设置当前瓦片的结束位置
}

/**
 * @brief 基于视锥体测试标记高斯为可见/不可见
 * 
 * 这是CudaRasterizer::Rasterizer类的成员函数，用于执行视锥体裁剪
 * 
 * @param P 高斯点的总数
 * @param means3D 3D高斯中心位置数组
 * @param viewmatrix 视图变换矩阵
 * @param projmatrix 投影变换矩阵
 * @param present 输出布尔数组，标记每个点是否可见
 */
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	// 启动CUDA核函数，使用256个线程的块大小
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

/**
 * @brief 从内存块创建几何状态对象
 * 
 * 从预分配的内存块中分配和初始化几何状态所需的所有缓冲区
 * 这包括深度、2D坐标、协方差矩阵、颜色等几何相关数据
 * 
 * @param chunk 内存块指针的引用，函数会更新这个指针
 * @param P 高斯点的数量
 * @return GeometryState 初始化完成的几何状态对象
 */
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);              // 分配深度缓冲区
	obtain(chunk, geom.clamped, P * 3, 128);         // 分配裁剪坐标缓冲区
	obtain(chunk, geom.internal_radii, P, 128);      // 分配内部半径缓冲区
	obtain(chunk, geom.means2D, P, 128);             // 分配2D中心点缓冲区
	obtain(chunk, geom.cov3D, P * 6, 128);           // 分配3D协方差矩阵缓冲区
	obtain(chunk, geom.conic_opacity, P, 128);       // 分配圆锥不透明度缓冲区
	obtain(chunk, geom.rgb, P * 3, 128);             // 分配RGB颜色缓冲区
	obtain(chunk, geom.normals, P * 3, 128);         // 分配法线向量缓冲区
	obtain(chunk, geom.tiles_touched, P, 128);       // 分配瓦片接触数缓冲区
	// 计算前缀和扫描所需的临时存储大小
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);  // 分配扫描临时空间
	obtain(chunk, geom.point_offsets, P, 128);       // 分配点偏移缓冲区
	return geom;
}

/**
 * @brief 从内存块创建图像状态对象
 * 
 * 从预分配的内存块中分配和初始化图像状态所需的所有缓冲区
 * 这包括累积透明度、贡献数量、瓦片范围等图像相关数据
 * 
 * @param chunk 内存块指针的引用，函数会更新这个指针
 * @param N 像素数量（通常是width * height）
 * @return ImageState 初始化完成的图像状态对象
 */
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);          // 分配累积透明度缓冲区
	obtain(chunk, img.n_contrib, N, 128);            // 分配贡献数量缓冲区
	obtain(chunk, img.ranges, N, 128);               // 分配瓦片范围缓冲区
	return img;
}

/**
 * @brief 从内存块创建分箱状态对象
 * 
 * 从预分配的内存块中分配和初始化分箱状态所需的所有缓冲区
 * 这包括点列表、键值对、排序空间等分箱和排序相关数据
 * 
 * @param chunk 内存块指针的引用，函数会更新这个指针
 * @param P 要处理的点数量
 * @return BinningState 初始化完成的分箱状态对象
 */
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);                // 分配排序后的点列表缓冲区
	obtain(chunk, binning.point_list_unsorted, P, 128);       // 分配未排序的点列表缓冲区
	obtain(chunk, binning.point_list_keys, P, 128);           // 分配排序后的键缓冲区
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);  // 分配未排序的键缓冲区
	// 计算基数排序所需的临时存储大小
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);  // 分配排序临时空间
	return binning;
}

/**
 * @brief 高斯可微分光栅化的前向渲染过程
 * 
 * 这是整个高斯光栅化系统的核心函数，执行从3D高斯到2D图像的完整渲染流程
 * 包括预处理、分箱、排序和最终的混合渲染
 * 
 * @param geometryBuffer 几何状态缓冲区分配函数
 * @param binningBuffer 分箱状态缓冲区分配函数
 * @param imageBuffer 图像状态缓冲区分配函数
 * @param P 高斯点的数量
 * @param D 球谐函数的度数
 * @param M 球谐函数的系数数量
 * @param background 背景颜色（RGB）
 * @param width 图像宽度
 * @param height 图像高度
 * @param means3D 3D高斯中心点坐标
 * @param shs 球谐函数系数
 * @param colors_precomp 预计算的颜色（可选）
 * @param opacities 不透明度值
 * @param scales 缩放参数
 * @param scale_modifier 缩放修饰符
 * @param rotations 旋转四元数
 * @param cov3D_precomp 预计算的3D协方差矩阵（可选）
 * @param viewmatrix 视图矩阵
 * @param projmatrix 投影矩阵
 * @param cam_pos 相机位置
 * @param tan_fovx X方向视场角的正切值
 * @param tan_fovy Y方向视场角的正切值
 * @param prefiltered 是否预过滤
 * @param out_color 输出颜色缓冲区
 * @param out_depth 输出深度缓冲区
 * @param out_normals 输出累积缓冲区	
 * @param out_opacity 输出不透明度缓冲区
 * @param radii 高斯半径缓冲区
 * @param is_used 使用标记缓冲区
 * @param debug 调试模式标志
 * @return int 实际渲染的高斯实例数量
 */
int CudaRasterizer::Rasterizer::forward(
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
	float* out_normals,
	float* out_opacity,
	int* radii,
	int* is_used,
	bool debug)
{
	// 根据视场角计算焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// 分配几何状态所需的内存并初始化
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// 如果没有提供半径缓冲区，使用内部半径缓冲区
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 计算瓦片网格和线程块的维度
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// 在训练过程中动态调整基于图像的辅助缓冲区大小
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// 对于非RGB通道，必须提供预计算的高斯颜色
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// 对每个高斯进行预处理（变换、边界计算、球谐函数转RGB、法线计算）
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.normals,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// 计算高斯接触瓦片数量的前缀和
	// 例如：[2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// 获取要渲染的高斯实例总数并调整辅助缓冲区大小
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// 分配分箱状态所需的内存
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// 为每个要渲染的实例生成适当的[瓦片|深度]键
	// 以及相应的重复高斯索引用于排序
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	// 计算排序所需的位数
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// 按键对完整的（重复的）高斯索引列表进行排序
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	// 初始化瓦片范围缓冲区
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// 在排序列表中识别每个瓦片工作负载的开始和结束位置
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// 让每个瓦片独立并行地混合其范围内的高斯
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.normals,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		out_depth,
		out_normals,
		out_opacity,
		is_used
		), debug)

	// 返回实际渲染的高斯实例数量
	return num_rendered;
}

/**
 * @brief 高斯可微分光栅化的反向传播过程
 * 
 * 计算优化所需的梯度，对应于前向渲染过程
 * 从像素级损失梯度反向传播到高斯参数梯度
 * 
 * @param P 高斯点的数量
 * @param D 球谐函数的度数
 * @param M 球谐函数的系数数量
 * @param R 渲染的高斯实例数量
 * @param background 背景颜色
 * @param width 图像宽度
 * @param height 图像高度
 * @param means3D 3D高斯中心点坐标
 * @param shs 球谐函数系数
 * @param colors_precomp 预计算的颜色（可选）
 * @param scales 缩放参数
 * @param scale_modifier 缩放修饰符
 * @param rotations 旋转四元数
 * @param cov3D_precomp 预计算的3D协方差矩阵（可选）
 * @param viewmatrix 视图矩阵
 * @param projmatrix 投影矩阵
 * @param projmatrix_raw 原始投影矩阵
 * @param campos 相机位置
 * @param tan_fovx X方向视场角的正切值
 * @param tan_fovy Y方向视场角的正切值
 * @param radii 高斯半径
 * @param geom_buffer 几何状态缓冲区
 * @param binning_buffer 分箱状态缓冲区
 * @param img_buffer 图像状态缓冲区
 * @param dL_dpix 像素颜色损失梯度
 * @param dL_dpix_depth 像素深度损失梯度
 * @param dL_dmean2D 2D中心点梯度输出
 * @param dL_dconic 圆锥矩阵梯度输出
 * @param dL_dopacity 不透明度梯度输出
 * @param dL_dcolor 颜色梯度输出
 * @param dL_ddepth 深度梯度输出
 * @param dL_dmean3D 3D中心点梯度输出
 * @param dL_dcov3D 3D协方差矩阵梯度输出
 * @param dL_dsh 球谐函数梯度输出
 * @param dL_dscale 缩放参数梯度输出
 * @param dL_drot 旋转参数梯度输出
 * @param dL_dtau tau参数梯度输出
 * @param debug 调试模式标志
 */
void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_dpix_depth,
	const float* dL_dpix_normal,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dnormals,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dtau,
	bool debug)
{
	// 从缓冲区恢复状态对象
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	// 如果没有提供半径，使用内部半径
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 计算焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// 设置瓦片网格和线程块维度
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// 从每像素损失梯度计算关于2D中心位置、圆锥矩阵、
	// 不透明度和高斯RGB的损失梯度
	// 如果给定了预计算颜色而不是球谐函数，则使用预计算颜色
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    const float* depth_ptr = geomState.depths;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		depth_ptr,
		geomState.normals,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_depth,
		dL_dpix_normal,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth,
		dL_dnormals
	), debug)

	// 处理预处理的其余部分。是给定了预计算的协方差
	// 还是缩放/旋转对？如果是预计算的，传递它；如果不是，
	// 使用我们自己计算的协方差
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
        projmatrix_raw,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		dL_dtau), debug)
}