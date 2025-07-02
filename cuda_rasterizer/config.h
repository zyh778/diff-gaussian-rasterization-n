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
 * @file config.h
 * @brief CUDA高斯光栅化器的配置参数定义
 * 
 * 该文件定义了光栅化器的核心配置参数，包括颜色通道数和CUDA线程块尺寸
 * 这些参数在编译时确定，影响整个光栅化系统的性能和功能
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

/**
 * @brief 颜色通道数量
 * 
 * 定义输出图像的颜色通道数量，默认为3（RGB）
 * 可以根据需要修改为其他值（如4用于RGBA）
 */
#define NUM_CHANNELS 3

/**
 * @brief CUDA线程块的X维度大小
 * 
 * 定义CUDA kernel中线程块在X方向的大小
 * 与BLOCK_Y一起决定了瓦片的尺寸（16x16像素）
 * 这个值影响GPU的占用率和性能
 */
#define BLOCK_X 16

/**
 * @brief CUDA线程块的Y维度大小
 * 
 * 定义CUDA kernel中线程块在Y方向的大小
 * 与BLOCK_X一起决定了瓦片的尺寸（16x16像素）
 * 这个值影响GPU的占用率和性能
 */
#define BLOCK_Y 16

#endif