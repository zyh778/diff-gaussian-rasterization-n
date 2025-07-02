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
 * @file auxiliary.h
 * @brief CUDA高斯光栅化器的辅助函数和常量定义
 * 
 * 该文件包含了光栅化过程中使用的各种辅助函数，包括：
 * - 球谐函数系数
 * - 坐标变换函数
 * - 数学运算函数
 * - 视锥体裁剪函数
 * - CUDA错误检查宏
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

/** @brief CUDA线程块的总大小（线程数） */
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

/** @brief 每个线程块中的warp数量（每个warp包含32个线程） */
#define NUM_WARPS (BLOCK_SIZE/32)

/**
 * @brief 球谐函数系数
 * 
 * 这些常量用于球谐函数的计算，将球谐函数系数转换为RGB颜色
 * 球谐函数用于表示依赖于视角的颜色变化
 */

/** @brief 0阶球谐函数系数 */
__device__ const float SH_C0 = 0.28209479177387814f;

/** @brief 1阶球谐函数系数 */
__device__ const float SH_C1 = 0.4886025119029199f;

/** @brief 2阶球谐函数系数数组 */
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};

/** @brief 3阶球谐函数系数数组 */
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

/**
 * @brief 将标准化设备坐标（NDC）转换为像素坐标
 * 
 * @param v NDC坐标值（范围[-1, 1]）
 * @param S 屏幕尺寸（宽度或高度）
 * @return float 对应的像素坐标
 */
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

/**
 * @brief 计算高斯覆盖的瓦片矩形范围
 * 
 * 根据高斯的中心位置和最大半径，计算其影响的瓦片范围
 * 用于确定需要处理该高斯的瓦片集合
 * 
 * @param p 高斯在屏幕空间的中心位置
 * @param max_radius 高斯的最大影响半径（像素）
 * @param rect_min 输出：覆盖范围的最小瓦片坐标
 * @param rect_max 输出：覆盖范围的最大瓦片坐标
 * @param grid 瓦片网格的维度
 */
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

/**
 * @brief 使用4x3矩阵变换3D点
 * 
 * 将3D点通过4x3变换矩阵进行变换，通常用于模型视图变换
 * 矩阵按列主序存储
 * 
 * @param p 待变换的3D点
 * @param matrix 4x3变换矩阵（按列主序存储）
 * @return float3 变换后的3D点
 */
__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

/**
 * @brief 使用4x4矩阵变换3D点
 * 
 * 将3D点通过4x4变换矩阵进行变换，返回齐次坐标
 * 通常用于投影变换，矩阵按列主序存储
 * 
 * @param p 待变换的3D点
 * @param matrix 4x4变换矩阵（按列主序存储）
 * @return float4 变换后的齐次坐标点
 */
__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

/**
 * @brief 使用4x3矩阵变换3D向量
 * 
 * 将3D向量通过4x3变换矩阵进行变换，不包含平移分量
 * 用于变换方向向量或法向量
 * 
 * @param p 待变换的3D向量
 * @param matrix 4x3变换矩阵（按列主序存储）
 * @return float3 变换后的3D向量
 */
__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

/**
 * @brief 使用4x3矩阵的转置变换3D向量
 * 
 * 将3D向量通过4x3变换矩阵的转置进行变换
 * 通常用于将向量从世界空间变换到局部空间
 * 
 * @param p 待变换的3D向量
 * @param matrix 4x3变换矩阵（按列主序存储）
 * @return float3 变换后的3D向量
 */
__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

/**
 * @brief 计算向量归一化对z分量的导数
 * 
 * 计算归一化向量的z分量相对于原向量变化的导数
 * 用于反向传播中计算梯度
 * 
 * @param v 原始向量
 * @param dv 向量的变化量
 * @return float 归一化向量z分量的导数
 */
__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

/**
 * @brief 计算向量归一化对各分量的导数
 * 
 * 计算归一化向量相对于原向量各分量变化的导数
 * 用于反向传播中计算完整的梯度向量
 * 
 * @param v 原始向量
 * @param dv 向量的变化量
 * @return float3 归一化向量各分量的导数
 */
__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

/**
 * @brief 计算四维向量归一化对各分量的导数
 * 
 * 计算归一化四维向量相对于原向量各分量变化的导数
 * 主要用于四元数归一化的梯度计算
 * 
 * @param v 原始四维向量
 * @param dv 四维向量的变化量
 * @return float4 归一化四维向量各分量的导数
 */
__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

/**
 * @brief Sigmoid激活函数
 * 
 * 计算sigmoid函数值，将输入映射到(0,1)范围
 * 常用于神经网络和概率计算
 * 
 * @param x 输入值
 * @return float sigmoid函数的输出值
 */
__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief 检查点是否在视锥体内
 * 
 * 判断给定的3D点是否在相机的视锥体范围内，用于剔除不可见的点
 * 主要检查点是否在相机前方（z > 0.2）
 * 
 * @param idx 点的索引
 * @param orig_points 原始点坐标数组
 * @param viewmatrix 视图变换矩阵
 * @param projmatrix 投影变换矩阵
 * @param prefiltered 是否已经预过滤
 * @param p_view 输出：点在视图空间的坐标
 * @return bool 点是否在视锥体内
 */
__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

/**
 * @brief 四元数乘法
 * 
 * 计算两个四元数的乘积，用于组合旋转
 * 四元数格式：(x, y, z, w)，其中w是实部
 * 
 * @param a 第一个四元数
 * @param b 第二个四元数
 * @return float4 两个四元数的乘积
 */
__forceinline__ __device__ float4 quat_mult(float4 a, float4 b)
{
	float4 q;
	q.x =  a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x;
	q.y = -a.x * b.z + a.y * b.w + a.z * b.x + a.w * b.y;
	q.z =  a.x * b.y - a.y * b.x + a.z * b.w + a.w * b.z;
	q.w = -a.x * b.x - a.y * b.y - a.z * b.z + a.w * b.w;
	return q;
}

/**
 * @brief CUDA错误检查宏
 * 
 * 执行CUDA操作并在调试模式下检查错误
 * 如果发生错误，输出错误信息并抛出异常
 * 
 * @param A 要执行的CUDA操作
 * @param debug 是否启用调试模式
 */
#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif