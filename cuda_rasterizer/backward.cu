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
 * @file backward.cu
 * @brief 高斯光栅化反向传播实现
 * 
 * 本文件实现了3D高斯光栅化的反向传播算法，包括：
 * - 球谐函数到RGB颜色转换的反向传播
 * - 2D协方差矩阵计算的反向传播
 * - 3D协方差矩阵计算的反向传播
 * - 预处理步骤的反向传播
 * - 渲染过程的反向传播
 * 
 * 这些函数计算损失函数相对于各种高斯参数的梯度，用于优化过程。
 */

#include "backward.h"
#include "auxiliary.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/**
 * @brief 球谐函数到RGB颜色转换的反向传播
 * 
 * 计算损失函数相对于球谐系数和高斯中心位置的梯度。
 * 球谐函数用于表示视角相关的颜色，需要根据观察方向计算RGB值。
 * 
 * @param idx 高斯索引
 * @param deg 球谐函数的度数（0, 1, 2, 3）
 * @param max_coeffs 最大球谐系数数量
 * @param means 高斯中心位置数组
 * @param campos 相机位置
 * @param shs 球谐系数数组
 * @param clamped 颜色是否被截断的标志数组
 * @param dL_dcolor 损失相对于颜色的梯度
 * @param dL_dmeans 输出：损失相对于高斯中心的梯度
 * @param dL_dshs 输出：损失相对于球谐系数的梯度
 * @param dL_dtau 输出：损失相对于相机参数的梯度
 */
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs,  float *dL_dtau)
{
	// 计算中间值，与前向传播过程相同
	glm::vec3 pos = means[idx];  // 当前高斯的3D位置
	glm::vec3 dir_orig = pos - campos;  // 从相机到高斯的原始方向向量
	glm::vec3 dir = dir_orig / glm::length(dir_orig);  // 归一化的观察方向

	// 获取当前高斯的球谐系数指针
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// 使用PyTorch的截断规则：如果应用了截断，梯度变为0
	// 这防止了梯度在颜色被截断到[0,1]范围时的传播
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;  // R通道
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;  // G通道
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;  // B通道

	// 初始化RGB相对于观察方向各分量的梯度
	glm::vec3 dRGBdx(0, 0, 0);  // RGB相对于x方向的梯度
	glm::vec3 dRGBdy(0, 0, 0);  // RGB相对于y方向的梯度
	glm::vec3 dRGBdz(0, 0, 0);  // RGB相对于z方向的梯度
	float x = dir.x;  // 观察方向的x分量
	float y = dir.y;  // 观察方向的y分量
	float z = dir.z;  // 观察方向的z分量

	// 当前高斯写入球谐梯度的目标位置
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// 球谐函数梯度计算，使用标准的微积分链式法则
	// 0阶球谐函数（常数项）
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	
	// 1阶球谐函数（线性项）
	if (deg > 0)
	{
		// 计算RGB相对于1阶球谐系数的梯度
		float dRGBdsh1 = -SH_C1 * y;  // Y_{1,-1}
		float dRGBdsh2 = SH_C1 * z;   // Y_{1,0}
		float dRGBdsh3 = -SH_C1 * x;  // Y_{1,1}
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		// 计算RGB相对于观察方向的梯度（1阶项贡献）
		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		// 2阶球谐函数（二次项）
		if (deg > 1)
		{
			// 预计算常用的二次项
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			// 计算RGB相对于2阶球谐系数的梯度
			float dRGBdsh4 = SH_C2[0] * xy;                    // Y_{2,-2}
			float dRGBdsh5 = SH_C2[1] * yz;                    // Y_{2,-1}
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);  // Y_{2,0}
			float dRGBdsh7 = SH_C2[3] * xz;                    // Y_{2,1}
			float dRGBdsh8 = SH_C2[4] * (xx - yy);             // Y_{2,2}
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			// 累加RGB相对于观察方向的梯度（2阶项贡献）
			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			// 3阶球谐函数（三次项）
			if (deg > 2)
			{
				// 计算RGB相对于3阶球谐系数的梯度
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);           // Y_{3,-3}
				float dRGBdsh10 = SH_C3[1] * xy * z;                       // Y_{3,-2}
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);     // Y_{3,-1}
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy); // Y_{3,0}
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);     // Y_{3,1}
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);                // Y_{3,2}
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);          // Y_{3,3}
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				// 累加RGB相对于观察方向的梯度（3阶项贡献）
				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// 观察方向是球谐函数计算的输入。观察方向受高斯中心位置影响，
	// 因此球谐函数的梯度必须传播回3D位置。
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// 考虑方向向量归一化的影响
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// 损失相对于高斯中心的梯度，但仅包括由于中心位置影响视角相关颜色的部分。
	// 额外的中心梯度将在后续方法中累加。
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
	// 相机参数的梯度（平移部分）
	dL_dtau[6 * idx + 0] += -dL_dmean.x;
	dL_dtau[6 * idx + 1] += -dL_dmean.y;
	dL_dtau[6 * idx + 2] += -dL_dmean.z;

}

/**
 * @brief 计算二维协方差矩阵反向传播的CUDA核函数
 * 
 * 该函数计算从三维协方差矩阵投影到二维屏幕空间协方差矩阵的反向传播梯度。
 * 由于计算复杂度较高，作为独立核函数在其他反向传播步骤之前启动。
 * 
 * @param P 高斯数量
 * @param means 高斯中心位置数组
 * @param radii 高斯在屏幕空间的半径数组
 * @param cov3Ds 三维协方差矩阵数组（每个高斯6个元素：上三角矩阵）
 * @param h_x 屏幕空间X方向的焦距参数
 * @param h_y 屏幕空间Y方向的焦距参数
 * @param tan_fovx X方向视场角的正切值
 * @param tan_fovy Y方向视场角的正切值
 * @param view_matrix 视图矩阵
 * @param dL_dconics 损失相对于二维协方差逆矩阵的梯度
 * @param dL_dmeans 输出：损失相对于高斯中心的梯度
 * @param dL_dcov 输出：损失相对于三维协方差矩阵的梯度
 * @param dL_dtau 输出：损失相对于相机参数的梯度
 */
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float *dL_dtau)
{
	// 获取当前线程对应的高斯索引
	auto idx = cg::this_grid().thread_rank();
	// 跳过无效的高斯（超出范围或半径为0）
	if (idx >= P || !(radii[idx] > 0))
		return;

	// 读取当前高斯的三维协方差矩阵位置
	const float* cov3D = cov3Ds + 6 * idx;

	// 获取梯度，重新计算二维协方差矩阵和反向传播所需的中间结果
	float3 mean = means[idx];  // 高斯中心位置
	// 提取二维协方差逆矩阵的梯度（对称矩阵的上三角部分）
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	// 将高斯中心变换到相机坐标系
	float3 t = transformPoint4x3(mean, view_matrix);
	
	// 视锥体裁剪限制
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	// 计算归一化设备坐标
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	// 应用视锥体裁剪
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	// 计算梯度乘数（用于处理裁剪边界的梯度）
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	// 构建雅可比矩阵J：从相机坐标到屏幕坐标的变换
	// J = [∂u/∂x  ∂u/∂y  ∂u/∂z]
	//     [∂v/∂x  ∂v/∂y  ∂v/∂z]
	//     [  0      0      0  ]
	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 提取视图矩阵的旋转部分W（3x3）
	// W将世界坐标变换到相机坐标
	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	// 从上三角存储格式重构三维协方差矩阵Vrk
	// cov3D存储格式：[σ_xx, σ_xy, σ_xz, σ_yy, σ_yz, σ_zz]
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 计算变换矩阵T = W * J（世界坐标到屏幕坐标的雅可比）
	glm::mat3 T = W * J;

	// 计算二维协方差矩阵：Σ_2D = T^T * Σ_3D * T
	// 这是高斯椭球在屏幕空间的投影
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// 使用辅助变量表示二维协方差矩阵元素，更紧凑
	// 添加正则化项0.3f以确保数值稳定性
	float a = cov2D[0][0] += 0.3f;  // σ_xx + 0.3
	float b = cov2D[0][1];          // σ_xy
	float c = cov2D[1][1] += 0.3f;  // σ_yy + 0.3

	float denom = a * c - b * b;    // 行列式
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);  // 1/det²，添加小值防止除零

	if (denom2inv != 0)
	{
		// 计算损失相对于二维协方差矩阵元素的梯度，
		// 给定损失相对于圆锥矩阵（协方差逆矩阵）的梯度
		// 使用逆矩阵求导公式：d(A^-1)/dA = -A^-1 * (dA) * A^-1
		// 例如：dL/da = dL/d_conic_a * d_conic_a/da
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// 计算损失相对于三维协方差矩阵(Vrk)每个元素的梯度（对角线元素），
		// 给定损失相对于二维协方差矩阵的梯度
		// 使用链式法则：cov2D = transpose(T) * transpose(Vrk) * T
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);  // σ_xx
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);  // σ_yy
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);  // σ_zz

		// 计算损失相对于三维协方差矩阵(Vrk)每个元素的梯度（非对角线元素），
		// 给定损失相对于二维协方差矩阵的梯度
		// 非对角线元素出现两次 --> 梯度需要乘以2
		// 使用链式法则：cov2D = transpose(T) * transpose(Vrk) * T
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;  // σ_xy
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;  // σ_xz
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;  // σ_yz
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// 计算损失相对于中间矩阵T上半部分(2x3)的梯度
	// 使用链式法则：cov2D = transpose(T) * transpose(Vrk) * T
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// 计算损失相对于雅可比矩阵J上半部分(3x2)非零元素的梯度
	// 使用链式法则：T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	// 预计算深度相关的倒数项
	float tz = 1.f / t.z;   // 1/z
	float tz2 = tz * tz;    // 1/z²
	float tz3 = tz2 * tz;   // 1/z³

	// 计算损失相对于变换后高斯中心t的梯度
	// 考虑视锥体裁剪对梯度的影响
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;  // ∂L/∂tx
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;  // ∂L/∂ty
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;  // ∂L/∂tz

	SE3 T_CW(view_matrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t_ = T_CW.t();
	mat33 dpC_drho = mat33::identity();
	mat33 dpC_dtheta = -mat33::skew_symmetric(t);
	float dL_dt[6];
	for (int i = 0; i < 3; i++) {
		float3 c_rho = dpC_drho.cols[i];
		float3 c_theta = dpC_dtheta.cols[i];
		dL_dt[i] = dL_dtx * c_rho.x + dL_dty * c_rho.y + dL_dtz * c_rho.z;
		dL_dt[i + 3] = dL_dtx * c_theta.x + dL_dty * c_theta.y + dL_dtz * c_theta.z;
	}
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// 计算损失相对于原始高斯中心位置的梯度
	// 考虑从世界坐标到相机坐标的变换：t = transformPoint4x3(mean, view_matrix)
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// 损失相对于高斯中心的梯度，但仅包括由于中心位置影响协方差矩阵的部分。
	// 额外的中心梯度将在BACKWARD::preprocess中累加。
	dL_dmeans[idx] = dL_dmean;

	// 计算损失相对于视图变换矩阵W的梯度
	// 使用链式法则：T = W * J
	float dL_dW00 = J[0][0] * dL_dT00;  // ∂L/∂W₀₀
	float dL_dW01 = J[0][0] * dL_dT01;  // ∂L/∂W₀₁
	float dL_dW02 = J[0][0] * dL_dT02;  // ∂L/∂W₀₂
	float dL_dW10 = J[1][1] * dL_dT10;  // ∂L/∂W₁₀
	float dL_dW11 = J[1][1] * dL_dT11;  // ∂L/∂W₁₁
	float dL_dW12 = J[1][1] * dL_dT12;  // ∂L/∂W₁₂
	float dL_dW20 = J[0][2] * dL_dT00 + J[1][2] * dL_dT10;  // ∂L/∂W₂₀
	float dL_dW21 = J[0][2] * dL_dT01 + J[1][2] * dL_dT11;  // ∂L/∂W₂₁
	float dL_dW22 = J[0][2] * dL_dT02 + J[1][2] * dL_dT12;  // ∂L/∂W₂₂

	// 提取旋转矩阵R的列向量
	float3 c1 = R.cols[0];  // R的第一列
	float3 c2 = R.cols[1];  // R的第二列
	float3 c3 = R.cols[2];  // R的第三列

	// 构建损失相对于W矩阵的梯度矩阵
	float dL_dW_data[9];
	dL_dW_data[0] = dL_dW00;  // W₀₀
	dL_dW_data[3] = dL_dW01;  // W₀₁
	dL_dW_data[6] = dL_dW02;  // W₀₂
	dL_dW_data[1] = dL_dW10;  // W₁₀
	dL_dW_data[4] = dL_dW11;  // W₁₁
	dL_dW_data[7] = dL_dW12;  // W₁₂
	dL_dW_data[2] = dL_dW20;  // W₂₀
	dL_dW_data[5] = dL_dW21;  // W₂₁
	dL_dW_data[8] = dL_dW22;  // W₂₂

	// 构建梯度矩阵并提取列向量
	mat33 dL_dW(dL_dW_data);
	float3 dL_dWc1 = dL_dW.cols[0];  // ∂L/∂W的第一列
	float3 dL_dWc2 = dL_dW.cols[1];  // ∂L/∂W的第二列
	float3 dL_dWc3 = dL_dW.cols[2];  // ∂L/∂W的第三列

	// 计算旋转参数的反对称矩阵（用于旋转梯度计算）
	mat33 n_W1_x = -mat33::skew_symmetric(c1);  // -[c1]×
	mat33 n_W2_x = -mat33::skew_symmetric(c2);  // -[c2]×
	mat33 n_W3_x = -mat33::skew_symmetric(c3);  // -[c3]×

	// 计算损失相对于旋转参数θ的梯度
	// 使用李群SO(3)的切空间表示
	float3 dL_dtheta = {};
	dL_dtheta.x = dot(dL_dWc1, n_W1_x.cols[0]) + dot(dL_dWc2, n_W2_x.cols[0]) +
				dot(dL_dWc3, n_W3_x.cols[0]);  // ∂L/∂θₓ
	dL_dtheta.y = dot(dL_dWc1, n_W1_x.cols[1]) + dot(dL_dWc2, n_W2_x.cols[1]) +
				dot(dL_dWc3, n_W3_x.cols[1]);  // ∂L/∂θᵧ
	dL_dtheta.z = dot(dL_dWc1, n_W1_x.cols[2]) + dot(dL_dWc2, n_W2_x.cols[2]) +
				dot(dL_dWc3, n_W3_x.cols[2]);  // ∂L/∂θᵤ

	// 累加到相机参数梯度中（旋转部分）
	dL_dtau[6 * idx + 3] += dL_dtheta.x;  // θₓ梯度
	dL_dtau[6 * idx + 4] += dL_dtheta.y;  // θᵧ梯度
	dL_dtau[6 * idx + 5] += dL_dtheta.z;  // θᵤ梯度

}

/**
 * @brief 计算三维协方差矩阵反向传播的设备函数
 * 
 * 该函数计算从尺度和旋转参数到三维协方差矩阵转换的反向传播梯度。
 * 三维协方差矩阵 Σ = R * S * S^T * R^T，其中R是旋转矩阵，S是尺度矩阵。
 * 
 * @param idx 高斯索引
 * @param scale 高斯的三维尺度参数
 * @param mod 尺度修正因子
 * @param rot 高斯的旋转四元数
 * @param dL_dcov3Ds 损失相对于三维协方差矩阵的梯度
 * @param dL_dscales 输出：损失相对于尺度参数的梯度
 * @param dL_drots 输出：损失相对于旋转参数的梯度
 */
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// 重新计算三维协方差矩阵计算的中间结果
	// 提取四元数分量
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;  // 四元数实部
	float x = q.y;  // 四元数虚部i
	float y = q.z;  // 四元数虚部j
	float z = q.w;  // 四元数虚部k

	// 从四元数构建旋转矩阵R
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// 构建尺度矩阵S
	glm::mat3 S = glm::mat3(1.0f);

	// 应用尺度修正因子
	glm::vec3 s = mod * scale;
	S[0][0] = s.x;  // X轴尺度
	S[1][1] = s.y;  // Y轴尺度
	S[2][2] = s.z;  // Z轴尺度

	// 计算中间矩阵M = S * R
	glm::mat3 M = S * R;

	// 获取当前高斯的协方差矩阵梯度指针
	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	// 从输入梯度中提取对称矩阵的分量
	// 对角线元素的梯度
	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	// 非对角线元素的梯度（因对称性需要乘以0.5）
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// 将协方差矩阵的逐元素梯度转换为矩阵形式
	// 协方差矩阵是对称的，所以非对角线元素需要特殊处理
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// 计算损失相对于中间矩阵M的梯度
	// 使用链式法则：dΣ/dM = 2M，所以 dL/dM = 2M * dL/dΣ
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// 计算损失相对于尺度参数的梯度
	// M = S * R，所以 dL/dS = R^T * (dL/dM)^T
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	// 为计算旋转梯度，需要将尺度因子应用到梯度矩阵
	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// 计算损失相对于归一化四元数的梯度
	// 使用四元数到旋转矩阵的导数公式
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// 计算损失相对于非归一化四元数的梯度
	// 这里直接使用归一化四元数的梯度，因为输入四元数已经是归一化的
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

/**
 * @brief 高斯光栅化预处理阶段的反向传播CUDA核函数
 * 
 * 该函数处理预处理步骤的反向传播，除了协方差计算和求逆（这些由之前的核函数处理）。
 * 主要包括：
 * - 屏幕空间投影的反向传播
 * - 深度计算的反向传播
 * - 球谐函数到RGB颜色转换的反向传播
 * - 三维协方差矩阵计算的反向传播
 * 
 * @tparam C 颜色通道数
 * @param P 高斯数量
 * @param D 球谐函数维度
 * @param M 最大球谐函数阶数
 * @param means 高斯中心位置数组
 * @param radii 高斯在屏幕空间的半径数组
 * @param shs 球谐函数系数数组
 * @param clamped 颜色是否被截断的标志数组
 * @param scales 高斯尺度参数数组
 * @param rotations 高斯旋转四元数数组
 * @param scale_modifier 尺度修正因子
 * @param viewmatrix 视图矩阵
 * @param proj 投影矩阵
 * @param proj_raw 原始投影矩阵
 * @param campos 相机位置
 * @param dL_dmean2D 损失相对于二维投影位置的梯度
 * @param dL_dmeans 输出：损失相对于高斯中心的梯度
 * @param dL_dcolor 输出：损失相对于颜色的梯度
 * @param dL_ddepth 输出：损失相对于深度的梯度
 * @param dL_dcov3D 输出：损失相对于三维协方差矩阵的梯度
 * @param dL_dsh 输出：损失相对于球谐函数系数的梯度
 * @param dL_dscale 输出：损失相对于尺度参数的梯度
 * @param dL_drot 输出：损失相对于旋转参数的梯度
 * @param dL_dtau 输出：损失相对于相机参数的梯度
 */
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float *viewmatrix,
	const float* proj,
	const float *proj_raw,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float *dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float *dL_dtau)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	float alpha = 1.0f * m_w;
	float beta = -m_hom.x * m_w * m_w;
	float gamma = -m_hom.y * m_w * m_w;

	float a = proj_raw[0];
	float b = proj_raw[5];
	float c = proj_raw[10];
	float d = proj_raw[14];
	float e = proj_raw[11];

	SE3 T_CW(viewmatrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t = T_CW.t();
	float3 p_C = T_CW * m;
	mat33 dp_C_d_rho = mat33::identity();
	mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);

	float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
	float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

	float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1; // x.T A = A.T x
	float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
	float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
	float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

	float2 dmean2D_dtau[6];
	dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
	dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
	dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
	dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
	dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
	dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

	dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
	dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
	dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
	dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
	dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
	dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

	float dL_dt[6];
	for (int i = 0; i < 6; i++) {
		dL_dt[i] = dL_dmean2D[idx].x * dmean2D_dtau[i].x + dL_dmean2D[idx].y * dmean2D_dtau[i].y;
	}
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// Compute gradient update due to computing depths
	// p_orig = m
	// p_view = transformPoint4x3(p_orig, viewmatrix);
	// depth = p_view.z;
	float dL_dpCz = dL_ddepth[idx];
	dL_dmeans[idx].x += dL_dpCz * viewmatrix[2];
	dL_dmeans[idx].y += dL_dpCz * viewmatrix[6];
	dL_dmeans[idx].z += dL_dpCz * viewmatrix[10];

	for (int i = 0; i < 3; i++) {
		float3 c_rho = dp_C_d_rho.cols[i];
		float3 c_theta = dp_C_d_theta.cols[i];
		dL_dtau[6 * idx + i] += dL_dpCz * c_rho.z;
		dL_dtau[6 * idx + i + 3] += dL_dpCz * c_theta.z;
	}


	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, dL_dtau);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

template <typename T>
__device__ void inline reduce_helper(int lane, int i, T *data) {
  if (lane < i) {
    data[lane] += data[lane + i];
  }
}

template <typename group_t, typename... Lists>
__device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
  int lane = g.thread_rank();
  g.sync();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    (...,
     reduce_helper(
         lane, i, lists)); // Fold expression: apply reduce_helper for each list
    g.sync();
  }
}


/**
 * @brief 高斯光栅化渲染过程的反向传播CUDA核函数
 * 
 * 该函数实现渲染过程的反向传播，从像素级别的损失梯度开始，
 * 反向计算每个高斯参数的梯度。主要步骤包括：
 * - 从后向前遍历影响每个像素的高斯
 * - 计算alpha混合的反向传播
 * - 累积各个高斯参数的梯度
 * 
 * @tparam C 颜色通道数
 * @param ranges 每个tile中高斯的范围
 * @param point_list 排序后的高斯索引列表
 * @param W 图像宽度
 * @param H 图像高度
 * @param bg_color 背景颜色
 * @param points_xy_image 高斯在屏幕空间的投影位置
 * @param conic_opacity 二维协方差逆矩阵和不透明度
 * @param colors 高斯颜色数组
 * @param depths 高斯深度数组
 * @param normals 高斯法线数组
 * @param final_Ts 最终透射率数组
 * @param n_contrib 每个像素的贡献高斯数量
 * @param dL_dpixels 损失相对于像素颜色的梯度
 * @param dL_dpixels_depth 损失相对于像素深度的梯度
 * @param dL_dpixels_normal 损失相对于像素法线的梯度
 * @param dL_dmean2D 输出：损失相对于二维投影位置的梯度
 * @param dL_dconic2D 输出：损失相对于二维协方差逆矩阵的梯度
 * @param dL_dopacity 输出：损失相对于不透明度的梯度
 * @param dL_dcolors 输出：损失相对于颜色的梯度
 * @param dL_ddepths 输出：损失相对于深度的梯度
 * @param dL_dnormals 输出：损失相对于法线的梯度
 */
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float3* __restrict__ normals,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_depth,
	const float* __restrict__ dL_dpixels_normal,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths,
	float3* __restrict__ dL_dnormals)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float3 collected_normals[BLOCK_SIZE];

	__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
	__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
	__shared__ float dL_ddepths_shared[BLOCK_SIZE];
	__shared__ float3 dL_dnormals_shared[BLOCK_SIZE];
	__shared__ float dL_dopacity_shared[BLOCK_SIZE];
	__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C] = { 0 };
	float accum_rec_depth = 0;
	float dL_dpixel_depth = 0;
	float3 accum_rec_normal = {0.0f, 0.0f, 0.0f};
	float3 dL_dpixel_normal = {0.0f, 0.0f, 0.0f};
	if (inside) {
		#pragma unroll
		for (int i = 0; i < C; i++) {
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		}
		dL_dpixel_depth = dL_dpixels_depth[pix_id];
		dL_dpixel_normal.x = dL_dpixels_normal[0 * H * W + pix_id];
		dL_dpixel_normal.y = dL_dpixels_normal[1 * H * W + pix_id];
		dL_dpixel_normal.z = dL_dpixels_normal[2 * H * W + pix_id];
	}

	float last_alpha = 0.f;
	float last_color[C] = { 0.f };
	float last_depth = 0.f;
	float3 last_normal = {0.0f, 0.0f, 0.0f};

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	__shared__ int skip_counter;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_xy[tid] = points_xy_image[coll_id];
			collected_conic_opacity[tid] = conic_opacity[coll_id];
			#pragma unroll
			for (int i = 0; i < C; i++) {
				collected_colors[i * BLOCK_SIZE + tid] = colors[coll_id * C + i];
				
			}
			collected_depths[tid] = depths[coll_id];
			collected_normals[tid] = normals[coll_id];
		}
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
			block.sync();
			if (tid == 0) {
				skip_counter = 0;
			}
			block.sync();

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			bool skip = done;
			contributor = done ? contributor : contributor - 1;
			skip |= contributor >= last_contributor;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			skip |= power > 0.0f;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			skip |= alpha < 1.0f / 255.0f;

			if (skip) {
				atomicAdd(&skip_counter, 1);
			}
			block.sync();
			if (skip_counter == BLOCK_SIZE) {
				continue;
			}


			T = skip ? T : T / (1.f - alpha);

			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			float local_dL_dcolors[3];
			#pragma unroll
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = skip ? last_color[ch] : c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
			}
			dL_dcolors_shared[tid].x = local_dL_dcolors[0];
			dL_dcolors_shared[tid].y = local_dL_dcolors[1];
			dL_dcolors_shared[tid].z = local_dL_dcolors[2];

			const float depth = collected_depths[j];
			accum_rec_depth = skip ? accum_rec_depth : last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			last_depth = skip ? last_depth : depth;
			dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;
			dL_ddepths_shared[tid] = skip ? 0.f : dchannel_dcolor * dL_dpixel_depth;

			// 法线梯度计算
            // 获取当前高斯的法线
            const float3 normal = collected_normals[j];

            // 计算累积重建法线
            // 使用alpha混合公式:new_accum = last_alpha * last_normal + (1-last_alpha) * old_accum
            // 如果skip为true则保持原值不变
            accum_rec_normal.x = skip ? accum_rec_normal.x : last_alpha * last_normal.x + (1.f - last_alpha) * accum_rec_normal.x;
            accum_rec_normal.y = skip ? accum_rec_normal.y : last_alpha * last_normal.y + (1.f - last_alpha) * accum_rec_normal.y; 
            accum_rec_normal.z = skip ? accum_rec_normal.z : last_alpha * last_normal.z + (1.f - last_alpha) * accum_rec_normal.z;

            // 更新last_normal用于下一次迭代
            last_normal = skip ? last_normal : normal;

            // 计算alpha的梯度
            // dL_dalpha += (normal - accum_rec_normal) * dL_dpixel_normal
            // 这是因为alpha影响了法线的混合权重
            dL_dalpha += (normal.x - accum_rec_normal.x) * dL_dpixel_normal.x;
            dL_dalpha += (normal.y - accum_rec_normal.y) * dL_dpixel_normal.y;
            dL_dalpha += (normal.z - accum_rec_normal.z) * dL_dpixel_normal.z;

            // 计算法线的梯度
            // dL_dnormal = dchannel_dcolor * dL_dpixel_normal
            // dchannel_dcolor是当前高斯对最终颜色的贡献权重
            dL_dnormals_shared[tid].x = skip ? 0.f : dchannel_dcolor * dL_dpixel_normal.x;
            dL_dnormals_shared[tid].y = skip ? 0.f : dchannel_dcolor * dL_dpixel_normal.y;
            dL_dnormals_shared[tid].z = skip ? 0.f : dchannel_dcolor * dL_dpixel_normal.z;
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0.f;
			#pragma unroll
			for (int i = 0; i < C; i++) {
				bg_dot_dpixel +=  bg_color[i] * dL_dpixel[i];
			}
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx * ddelx_dx;
			dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely * ddely_dy;
			dL_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * dL_dG;
			dL_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * dL_dG;
			dL_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * dL_dG;
			dL_dopacity_shared[tid] = skip ? 0.f : G * dL_dalpha;

			render_cuda_reduce_sum(block, 
				dL_dmean2D_shared,
				dL_dconic2D_shared,
				dL_dopacity_shared,
				dL_dcolors_shared, 
				dL_ddepths_shared,
				dL_dnormals_shared
			);	
			
			if (tid == 0) {
				float2 dL_dmean2D_acc = dL_dmean2D_shared[0];
				float4 dL_dconic2D_acc = dL_dconic2D_shared[0];
				float dL_dopacity_acc = dL_dopacity_shared[0];
				float3 dL_dcolors_acc = dL_dcolors_shared[0];
				float dL_ddepths_acc = dL_ddepths_shared[0];
				float3 dL_dnormals_acc = dL_dnormals_shared[0];

				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_acc.x);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_acc.x);
				atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_acc.w);
				atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);
				atomicAdd(&dL_dcolors[global_id * C + 0], dL_dcolors_acc.x);
				atomicAdd(&dL_dcolors[global_id * C + 1], dL_dcolors_acc.y);
				atomicAdd(&dL_dcolors[global_id * C + 2], dL_dcolors_acc.z);
				atomicAdd(&dL_ddepths[global_id], dL_ddepths_acc);
				atomicAdd(&dL_dnormals[global_id].x, dL_dnormals_acc.x);
				atomicAdd(&dL_dnormals[global_id].y, dL_dnormals_acc.y);
				atomicAdd(&dL_dnormals[global_id].z, dL_dnormals_acc.z);
			}
		}
	}
}

/**
 * @brief 高斯光栅化预处理阶段反向传播的主函数
 * 
 * 该函数执行预处理步骤的反向传播，包括：
 * - 二维协方差矩阵计算的反向传播
 * - 屏幕空间投影的反向传播
 * - 球谐函数到RGB颜色转换的反向传播
 * - 三维协方差矩阵计算的反向传播
 * 
 * @param P 高斯数量
 * @param D 球谐函数维度
 * @param M 最大球谐函数阶数
 * @param means3D 高斯中心位置数组
 * @param radii 高斯在屏幕空间的半径数组
 * @param shs 球谐函数系数数组
 * @param clamped 颜色是否被截断的标志数组
 * @param scales 高斯尺度参数数组
 * @param rotations 高斯旋转四元数数组
 * @param scale_modifier 尺度修正因子
 * @param cov3Ds 三维协方差矩阵数组
 * @param viewmatrix 视图矩阵
 * @param projmatrix 投影矩阵
 * @param projmatrix_raw 原始投影矩阵
 * @param focal_x X方向焦距
 * @param focal_y Y方向焦距
 * @param tan_fovx X方向视场角正切值
 * @param tan_fovy Y方向视场角正切值
 * @param campos 相机位置
 * @param dL_dmean2D 损失相对于二维投影位置的梯度
 * @param dL_dconic 损失相对于二维协方差逆矩阵的梯度
 * @param dL_dmean3D 输出：损失相对于三维中心位置的梯度
 * @param dL_dcolor 输出：损失相对于颜色的梯度
 * @param dL_ddepth 输出：损失相对于深度的梯度
 * @param dL_dcov3D 输出：损失相对于三维协方差矩阵的梯度
 * @param dL_dsh 输出：损失相对于球谐函数系数的梯度
 * @param dL_dscale 输出：损失相对于尺度参数的梯度
 * @param dL_drot 输出：损失相对于旋转参数的梯度
 * @param dL_dtau 输出：损失相对于相机参数的梯度
 */
void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_raw,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dtau)
{
	// 传播二维圆锥矩阵计算路径的梯度。
	// 由于计算较长，因此作为独立的核函数而不是"preprocess"的一部分。
	// 完成后，相对于3D均值的损失梯度已被修改，相对于3D协方差矩阵的梯度已被计算。
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dtau);

	// 传播剩余步骤的梯度：完成3D均值梯度，
	// 将颜色梯度传播到球谐函数（如果需要），将3D协方差
	// 矩阵梯度传播到尺度和旋转。
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		projmatrix_raw,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dtau);
}

/**
 * @brief 高斯光栅化渲染过程反向传播的主函数
 * 
 * 该函数启动CUDA核函数来执行渲染过程的反向传播，
 * 从像素级别的损失梯度计算各个高斯参数的梯度。
 * 
 * @param grid CUDA网格维度
 * @param block CUDA块维度
 * @param ranges 每个tile中高斯的范围
 * @param point_list 排序后的高斯索引列表
 * @param W 图像宽度
 * @param H 图像高度
 * @param bg_color 背景颜色
 * @param means2D 高斯在屏幕空间的投影位置
 * @param conic_opacity 二维协方差逆矩阵和不透明度
 * @param colors 高斯颜色数组
 * @param depths 高斯深度数组
 * @param final_Ts 最终透射率数组
 * @param n_contrib 每个像素的贡献高斯数量
 * @param dL_dpixels 损失相对于像素颜色的梯度
 * @param dL_dpixels_depth 损失相对于像素深度的梯度
 * @param dL_dmean2D 输出：损失相对于二维投影位置的梯度
 * @param dL_dconic2D 输出：损失相对于二维协方差逆矩阵的梯度
 * @param dL_dopacity 输出：损失相对于不透明度的梯度
 * @param dL_dcolors 输出：损失相对于颜色的梯度
 * @param dL_ddepths 输出：损失相对于深度的梯度
 */
void BACKWARD::render(
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
	float3* dL_dnormals)
{
	// 启动渲染反向传播CUDA核函数
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		normals,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixels_depth,
		dL_dpixels_normal,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths,
		dL_dnormals
		);
}