#include "utils.h"
#include "auxiliary.h"

// 实现论文 "3D Gaussian Splatting as Markov Chain Monte Carlo" 中的公式(9)
// 这个CUDA核函数用于计算高斯点的重定位
__global__ void compute_relocation(
    int P,                // 高斯点的总数
    float* opacity_old,   // 原始不透明度数组
    float* scale_old,     // 原始尺度数组
    int* N,              // 每个高斯点的分裂数量
    float* binoms,       // 预计算的二项式系数
    int n_max,           // 二项式系数的最大阶数
    float* opacity_new,  // 输出的新不透明度数组
    float* scale_new)    // 输出的新尺度数组
{
    // 计算当前线程处理的高斯点索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= P) return;
    
    int N_idx = N[idx];  // 获取当前高斯点的分裂数量
    float denom_sum = 0.0f;  // 用于累积分母和

    // 计算新的不透明度
    // 根据论文公式，新的不透明度为：α_new = 1 - (1 - α_old)^(1/N)
    opacity_new[idx] = 1.0f - powf(1.0f - opacity_old[idx], 1.0f / N_idx);
    
    // 计算新的尺度
    // 实现论文中的分母计算部分
    for (int i = 1; i <= N_idx; ++i) {
        for (int k = 0; k <= (i-1); ++k) {
            // 获取预计算的二项式系数
            float bin_coeff = binoms[(i-1) * n_max + k];
            // 计算每一项：(-1)^k / sqrt(k+1) * α_new^(k+1)
            float term = (pow(-1, k) / sqrt(k + 1)) * pow(opacity_new[idx], k + 1);
            denom_sum += (bin_coeff * term);
        }
    }

    // 计算最终的缩放系数
    float coeff = (opacity_old[idx] / denom_sum);
    
    // 更新三个维度的尺度值
    for (int i = 0; i < 3; ++i)
        scale_new[idx * 3 + i] = coeff * scale_old[idx * 3 + i];
}

// UTILS类中的包装函数，用于启动CUDA核函数
void UTILS::ComputeRelocation(
    int P,               // 高斯点的总数
    float* opacity_old,  // 原始不透明度数组
    float* scale_old,    // 原始尺度数组
    int* N,             // 每个高斯点的分裂数量
    float* binoms,      // 预计算的二项式系数
    int n_max,          // 二项式系数的最大阶数
    float* opacity_new, // 输出的新不透明度数组
    float* scale_new)   // 输出的新尺度数组
{
    // 计算需要的CUDA块数量，每个块256个线程
    int num_blocks = (P + 255) / 256;
    dim3 block(256, 1, 1);
    dim3 grid(num_blocks, 1, 1);
    // 启动CUDA核函数
    compute_relocation<<<grid, block>>>(P, opacity_old, scale_old, N, binoms, n_max, opacity_new, scale_new);
}