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

#include "forward.h"
#include "auxiliary.h"
#include "helper_math.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// 将球谐函数系数转换为RGB颜色的前向传播函数
// idx: 高斯体索引
// deg: 球谐函数的度数
// max_coeffs: 最大系数数量
// means: 高斯体中心点位置
// campos: 相机位置
// shs: 球谐函数系数
// clamped: 记录是否被截断到0的标志
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// 基于Zhang等人2022年论文"Differentiable Point-Based Radiance Fields for Efficient View Synthesis"的实现
	glm::vec3 pos = means[idx];
	// 计算从相机到高斯体的方向向量并归一化
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	// 获取当前高斯体的球谐系数
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	// 计算0阶球谐函数的贡献
	glm::vec3 result = SH_C0 * sh[0];

	// 计算更高阶球谐函数的贡献
	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		// 1阶球谐函数
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			// 2阶球谐函数
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				// 3阶球谐函数
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// 将RGB颜色限制在正值范围内,并记录是否被截断
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// 计算2D协方差矩阵的前向传播函数
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// 基于Zwicker等人2002年论文"EWA Splatting"中的公式29和31
	// 同时考虑了视口的缩放比例
	// 将点变换到相机空间
	float3 t = transformPoint4x3(mean, viewmatrix);

	// 限制投影点在视锥体内
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 计算雅可比矩阵J
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 视图矩阵W
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	// 计算变换矩阵T
	glm::mat3 T = W * J;

	// 构建3D协方差矩阵
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 计算2D协方差矩阵
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// 应用低通滤波:确保每个高斯体至少有1个像素的宽度/高度
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// 计算3D协方差矩阵的前向传播函数
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// 创建缩放矩阵
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// 获取四元数参数
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// 从四元数计算旋转矩阵
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// 计算变换矩阵M
	glm::mat3 M = S * R;

	// 计算3D世界空间协方差矩阵Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// 由于协方差矩阵是对称的,只存储上三角部分
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// 光栅化前的预处理CUDA核函数
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float* normals,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	// 获取当前线程的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// 初始化半径和触及的瓦片数为0
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// 执行近平面剔除
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// 将点投影到屏幕空间
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// 获取或计算3D协方差矩阵
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// 计算2D屏幕空间协方差矩阵
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// 计算协方差矩阵的逆(EWA算法)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// 计算屏幕空间范围并确定边界矩形
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// 计算或使用预计算的颜色
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// 计算法线向量
	// 1. 归一化四元数
	glm::vec4 quat = rotations[idx];
	float quat_norm = sqrt(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);
	quat = quat / quat_norm;
	
	// 2. 找到最小尺度轴（对应法线方向）
	glm::vec3 scale = scales[idx];
	int min_axis = 0;
	if (scale.y < scale.x) min_axis = 1;
	if (scale.z < scale[min_axis]) min_axis = 2;
	
	// 3. 创建单位向量（one-hot编码）
	float3 unit_normal = {0.0f, 0.0f, 0.0f};
	if (min_axis == 0) unit_normal.x = 1.0f;
	else if (min_axis == 1) unit_normal.y = 1.0f;
	else unit_normal.z = 1.0f;
	
	// 4. 将四元数转换为旋转矩阵并应用到法线
	float w = quat.w, x = quat.x, y = quat.y, z = quat.z;
	// 旋转矩阵的第一行、第二行、第三行
	float3 rotated_normal;
	rotated_normal.x = (1 - 2*y*y - 2*z*z) * unit_normal.x + (2*x*y - 2*w*z) * unit_normal.y + (2*x*z + 2*w*y) * unit_normal.z;
	rotated_normal.y = (2*x*y + 2*w*z) * unit_normal.x + (1 - 2*x*x - 2*z*z) * unit_normal.y + (2*y*z - 2*w*x) * unit_normal.z;
	rotated_normal.z = (2*x*z - 2*w*y) * unit_normal.x + (2*y*z + 2*w*x) * unit_normal.y + (1 - 2*x*x - 2*y*y) * unit_normal.z;
	
	// 5. 归一化法线
	float normal_norm = sqrt(rotated_normal.x * rotated_normal.x + rotated_normal.y * rotated_normal.y + rotated_normal.z * rotated_normal.z);
	rotated_normal.x /= normal_norm;
	rotated_normal.y /= normal_norm;
	rotated_normal.z /= normal_norm;
	
	// 6. 计算视线方向
	float3 viewdir;
	// 计算从高斯体中心点指向相机的视线方向向量
	// 通过相机位置减去高斯体位置得到视线向量的三个分量
	viewdir.x = cam_pos->x - p_orig.x;
	viewdir.y = cam_pos->y - p_orig.y; 
	viewdir.z = cam_pos->z - p_orig.z;
	float viewdir_norm = sqrt(viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z);
	viewdir.x /= viewdir_norm;
	viewdir.y /= viewdir_norm;
	viewdir.z /= viewdir_norm;
	
	// 7. 检查法线与视线方向的点积，如果为负则翻转法线
	float dot_product = rotated_normal.x * viewdir.x + rotated_normal.y * viewdir.y + rotated_normal.z * viewdir.z;
	if (dot_product < 0.0f) {
		rotated_normal.x = -rotated_normal.x;
		rotated_normal.y = -rotated_normal.y;
		rotated_normal.z = -rotated_normal.z;
	}
	
	// 8. 应用相机到世界的变换矩阵（viewmatrix的逆的转置，这里直接用viewmatrix的前3x3部分）
	// 注意：由于viewmatrix = world_view_transform = getWorld2View2(R, T).transpose(0, 1)
	// 而getWorld2View2构建的是世界到相机的变换，其转置后变成了相机到世界的变换
	// 因此这里正确的转换方式是直接使用viewmatrix的元素，但按照列主序排列
	float3 world_normal;
	world_normal.x = viewmatrix[0] * rotated_normal.x + viewmatrix[4] * rotated_normal.y + viewmatrix[8] * rotated_normal.z;
	world_normal.y = viewmatrix[1] * rotated_normal.x + viewmatrix[5] * rotated_normal.y + viewmatrix[9] * rotated_normal.z;
	world_normal.z = viewmatrix[2] * rotated_normal.x + viewmatrix[6] * rotated_normal.y + viewmatrix[10] * rotated_normal.z;
	
	// 9. 存储计算得到的法线
	normals[idx * 3 + 0] = world_normal.x;
	normals[idx * 3 + 1] = world_normal.y;
	normals[idx * 3 + 2] = world_normal.z;

	// 存储辅助数据供后续使用
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// 将2D协方差矩阵的逆和不透明度打包到一个float4中
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// 主光栅化CUDA核函数
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ normals,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	float* __restrict__ out_depth, 
	float* __restrict__ out_normals,
	float* __restrict__ out_opacity,
	int * __restrict__ is_used)
{
	// 获取当前线程块和像素信息
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// 检查线程是否对应有效像素
	bool inside = pix.x < W&& pix.y < H;
	bool done = !inside;

	// 加载需要处理的ID范围
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// 分配共享内存用于批量数据获取
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];

	// 初始化辅助变量
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0.0f;
	float N[3] = { 0.0f, 0.0f, 0.0f }; // 累积法线

	// 迭代处理所有批次
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// 如果整个块都完成了则退出
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// 集体将高斯体数据从全局内存加载到共享内存
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];
		}
		block.sync();

		// 处理当前批次中的每个高斯体
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

			// 使用圆锥矩阵重采样
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// 计算alpha值
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) {
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// 累积颜色、深度和法线值
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}
			for (int ch = 0; ch < 3; ch++) {
				N[ch] += normals[collected_id[j] * 3 + ch] * alpha * T;
			}
			D += collected_depth[j] * alpha * T;
			// 记录高斯体被多少像素使用
			if (test_T > 0.5f) {
				atomicAdd(&(is_used[collected_id[j]]), 1);
			}

			T = test_T;
			last_contributor = contributor;
		}
	}

	// 将最终渲染结果写入输出缓冲区
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_depth[pix_id] = D;
		// 输出法线（归一化后）
		float normal_length = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
		if (normal_length > 0.0f) {
			out_normals[0 * H * W + pix_id] = N[0] / normal_length;
			out_normals[1 * H * W + pix_id] = N[1] / normal_length;
			out_normals[2 * H * W + pix_id] = N[2] / normal_length;
		} else {
			out_normals[0 * H * W + pix_id] = 0.0f;
			out_normals[1 * H * W + pix_id] = 0.0f;
			out_normals[2 * H * W + pix_id] = 1.0f; // 默认向上法线
		}
		out_opacity[pix_id] = 1 - T;
	}
}

// 渲染函数的CPU端包装器
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
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
	int* is_used)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		normals,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth,
		out_depth,
		out_normals,
		out_opacity,
		is_used);
}

// 预处理函数的CPU端包装器
void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float* normals,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		normals,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}