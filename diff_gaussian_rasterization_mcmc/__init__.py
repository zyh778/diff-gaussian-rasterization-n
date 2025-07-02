#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
可微分高斯光栅化MCMC版本的Python接口模块

本模块提供了3D高斯溅射渲染的Python接口，包括：
- 高斯光栅化的前向和反向传播
- 可见性标记功能
- 重定位计算功能
- 光栅化设置和渲染器类

主要组件：
- GaussianRasterizer: 主要的渲染器类
- GaussianRasterizationSettings: 渲染设置的数据结构
- rasterize_gaussians: 核心光栅化函数
- compute_relocation: 重定位计算函数
"""

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C  # 导入编译的C++/CUDA扩展模块


def cpu_deep_copy_tuple(input_tuple):
    """
    深度复制元组中的张量到CPU
    
    用于调试模式下保存参数的副本，防止在GPU计算过程中被破坏
    
    Args:
        input_tuple: 包含张量和其他对象的元组
        
    Returns:
        tuple: 复制到CPU的元组，其中张量被克隆到CPU，其他对象保持不变
    """
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
):
    """
    高斯光栅化的主要接口函数
    
    将3D高斯椭球投影到2D图像平面并进行光栅化渲染
    
    Args:
        means3D: 3D高斯中心位置 [N, 3]
        means2D: 2D投影位置 [N, 2]
        sh: 球谐函数系数 [N, K, 3] (K取决于sh_degree)
        colors_precomp: 预计算的颜色 [N, 3]
        opacities: 不透明度 [N, 1]
        scales: 缩放参数 [N, 3]
        rotations: 旋转四元数 [N, 4]
        cov3Ds_precomp: 预计算的3D协方差矩阵 [N, 6]
        theta: MCMC相关的theta参数
        rho: MCMC相关的rho参数
        raster_settings: 光栅化设置对象
        
    Returns:
        tuple: (color, radii, depth, normal, opacity, is_used)
            - color: 渲染的颜色图像
            - radii: 每个高斯的半径
            - depth: 深度图
            - normal: 法向量图
            - opacity: 不透明度图
            - is_used: 是否被使用的标记
    """
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    """
    高斯光栅化的自动微分函数
    
    实现了高斯光栅化的前向传播和反向传播，支持梯度计算
    继承自torch.autograd.Function以支持自定义的前向和反向传播
    """
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    ):
        """
        前向传播：执行高斯光栅化渲染
        
        Args:
            ctx: PyTorch自动微分上下文，用于保存反向传播所需的信息
            其他参数同rasterize_gaussians函数
            
        Returns:
            tuple: 渲染结果 (color, radii, depth, normal, opacity, is_used)
        """

        # 按照C++库期望的方式重新组织参数
        args = (
            raster_settings.bg,              # 背景颜色
            means3D,                         # 3D高斯中心位置
            colors_precomp,                  # 预计算的颜色
            opacities,                       # 不透明度
            scales,                          # 缩放参数
            rotations,                       # 旋转四元数
            raster_settings.scale_modifier,  # 缩放修饰符
            cov3Ds_precomp,                 # 预计算的3D协方差矩阵
            raster_settings.viewmatrix,      # 视图矩阵
            raster_settings.projmatrix,      # 投影矩阵
            raster_settings.projmatrix_raw,  # 原始投影矩阵
            raster_settings.tanfovx,         # X方向视场角的正切值
            raster_settings.tanfovy,         # Y方向视场角的正切值
            raster_settings.image_height,    # 图像高度
            raster_settings.image_width,     # 图像宽度
            sh,                              # 球谐函数系数
            raster_settings.sh_degree,       # 球谐函数阶数
            raster_settings.campos,          # 相机位置
            raster_settings.prefiltered,     # 是否预过滤
            raster_settings.debug,           # 调试模式
        )

        # 调用C++/CUDA光栅化器
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # 在参数可能被破坏之前复制它们
            try:
                num_rendered, color, normal, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, is_used = (
                    _C.rasterize_gaussians(*args)
                )
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\n前向传播中发生错误。请转发snapshot_fw.dump文件进行调试。")
                raise ex
        else:
            num_rendered, color, normal, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, is_used = (
                _C.rasterize_gaussians(*args)
            )

        # 保存反向传播所需的相关张量
        ctx.raster_settings = raster_settings  # 保存光栅化设置
        ctx.num_rendered = num_rendered         # 保存渲染的高斯数量
        ctx.save_for_backward(
            colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer
        )
        return color, radii, depth, normal, opacity, is_used

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_normal,grad_out_opacity, grad_is_used):
        """
        反向传播：计算高斯光栅化的梯度
        
        Args:
            ctx: PyTorch自动微分上下文，包含前向传播保存的信息
            grad_out_color: 颜色输出的梯度
            grad_out_radii: 半径输出的梯度
            grad_out_depth: 深度输出的梯度
            grad_out_normal: 法向量输出的梯度
            grad_out_opacity: 不透明度输出的梯度
            grad_is_used: 使用标记的梯度
            
        Returns:
            tuple: 输入参数的梯度
        """

        # 从上下文中恢复必要的值
        num_rendered = ctx.num_rendered        # 渲染的高斯数量
        raster_settings = ctx.raster_settings  # 光栅化设置
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = (
            ctx.saved_tensors  # 前向传播保存的张量
        )

        # 按照C++方法期望的方式重新组织参数
        args = (
            raster_settings.bg,              # 背景颜色
            means3D,                         # 3D高斯中心位置
            radii,                           # 高斯半径
            colors_precomp,                  # 预计算的颜色
            scales,                          # 缩放参数
            rotations,                       # 旋转四元数
            raster_settings.scale_modifier,  # 缩放修饰符
            cov3Ds_precomp,                 # 预计算的3D协方差矩阵
            raster_settings.viewmatrix,      # 视图矩阵
            raster_settings.projmatrix,      # 投影矩阵
            raster_settings.projmatrix_raw,  # 原始投影矩阵
            raster_settings.tanfovx,         # X方向视场角的正切值
            raster_settings.tanfovy,         # Y方向视场角的正切值
            grad_out_color,                  # 颜色输出的梯度
            grad_out_depth,                  # 深度输出的梯度
            grad_out_normal,                 # 法向量输出的梯度
            sh,                              # 球谐函数系数
            raster_settings.sh_degree,       # 球谐函数阶数
            raster_settings.campos,          # 相机位置
            geomBuffer,                      # 几何缓冲区
            num_rendered,                    # 渲染的高斯数量
            binningBuffer,                   # 分箱缓冲区
            imgBuffer,                       # 图像缓冲区
            raster_settings.debug,           # 调试模式
        )

        # 通过调用反向传播方法计算相关张量的梯度
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # 在参数可能被破坏之前复制它们
            try:
                (
                    grad_means2D,        # 2D位置的梯度
                    grad_colors_precomp, # 预计算颜色的梯度
                    grad_opacities,      # 不透明度的梯度
                    grad_means3D,        # 3D位置的梯度
                    grad_cov3Ds_precomp, # 3D协方差的梯度
                    grad_sh,             # 球谐函数的梯度
                    grad_scales,         # 缩放参数的梯度
                    grad_rotations,      # 旋转参数的梯度
                    grad_tau,            # tau参数的梯度
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\n反向传播中发生错误。正在写入snapshot_bw.dump文件进行调试。\n")
                raise ex
        else:
            (
                grad_means2D,        # 2D位置的梯度
                grad_colors_precomp, # 预计算颜色的梯度
                grad_opacities,      # 不透明度的梯度
                grad_means3D,        # 3D位置的梯度
                grad_cov3Ds_precomp, # 3D协方差的梯度
                grad_sh,             # 球谐函数的梯度
                grad_scales,         # 缩放参数的梯度
                grad_rotations,      # 旋转参数的梯度
                grad_tau,            # tau参数的梯度
            ) = _C.rasterize_gaussians_backward(*args)

        # 处理tau梯度，分离为rho和theta的梯度
        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)  # 对tau梯度求和
        grad_rho = grad_tau[:3].view(1, -1)                # 提取rho的梯度（前3个元素）
        grad_theta = grad_tau[3:].view(1, -1)              # 提取theta的梯度（后3个元素）

        # 组织所有输入参数的梯度
        grads = (
            grad_means3D,        # 3D位置的梯度
            grad_means2D,        # 2D位置的梯度
            grad_sh,             # 球谐函数的梯度
            grad_colors_precomp, # 预计算颜色的梯度
            grad_opacities,      # 不透明度的梯度
            grad_scales,         # 缩放参数的梯度
            grad_rotations,      # 旋转参数的梯度
            grad_cov3Ds_precomp, # 3D协方差的梯度
            grad_theta,          # theta参数的梯度
            grad_rho,            # rho参数的梯度
            None,                # raster_settings不需要梯度
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    """
    高斯光栅化设置的数据结构
    
    使用NamedTuple定义光栅化过程中需要的所有参数和设置
    这些设置控制着渲染的各个方面，包括图像尺寸、相机参数、渲染选项等
    
    Attributes:
        image_height: 输出图像的高度（像素）
        image_width: 输出图像的宽度（像素）
        tanfovx: X方向视场角的正切值
        tanfovy: Y方向视场角的正切值
        bg: 背景颜色张量 [3]
        scale_modifier: 缩放修饰符，用于调整高斯的大小
        viewmatrix: 视图变换矩阵 [4, 4]
        projmatrix: 投影变换矩阵 [4, 4]
        projmatrix_raw: 原始投影矩阵 [4, 4]
        sh_degree: 球谐函数的阶数
        campos: 相机在世界坐标系中的位置 [3]
        prefiltered: 是否使用预过滤
        debug: 是否启用调试模式
    """
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    projmatrix_raw: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    """
    高斯光栅化器主类
    
    这是3D高斯溅射渲染的主要接口类，继承自nn.Module以支持PyTorch的自动微分
    封装了高斯光栅化的所有功能，包括渲染、可见性检测等
    
    Args:
        raster_settings: GaussianRasterizationSettings对象，包含渲染所需的所有设置
    """
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings  # 保存光栅化设置

    def markVisible(self, positions):
        """
        标记可见的高斯点
        
        基于相机视锥体裁剪来确定哪些3D高斯点在当前视角下是可见的
        
        Args:
            positions: 3D高斯中心位置 [N, 3]
            
        Returns:
            torch.Tensor: 布尔张量，标记每个点是否可见 [N]
        """
        # 使用视锥体裁剪标记可见点（基于相机的视锥体裁剪）
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(positions, raster_settings.viewmatrix, raster_settings.projmatrix)

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        theta=None,
        rho=None,
    ):
        """
        执行高斯光栅化渲染的前向传播
        
        Args:
            means3D: 3D高斯中心位置 [N, 3]
            means2D: 2D投影位置 [N, 2]
            opacities: 不透明度 [N, 1]
            shs: 球谐函数系数 [N, K, 3]，与colors_precomp二选一
            colors_precomp: 预计算的颜色 [N, 3]，与shs二选一
            scales: 缩放参数 [N, 3]，与cov3D_precomp二选一
            rotations: 旋转四元数 [N, 4]，与cov3D_precomp二选一
            cov3D_precomp: 预计算的3D协方差矩阵 [N, 6]，与scales/rotations二选一
            theta: MCMC相关的theta参数
            rho: MCMC相关的rho参数
            
        Returns:
            tuple: 渲染结果 (color, radii, depth, normal, opacity, is_used)
        """

        raster_settings = self.raster_settings

        # 验证颜色参数：必须提供球谐函数系数或预计算颜色中的一个
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception("请提供球谐函数系数(SHs)或预计算颜色(precomputed colors)中的一个！")

        # 验证几何参数：必须提供缩放/旋转对或预计算3D协方差中的一个
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception("请提供缩放/旋转参数对或预计算3D协方差矩阵中的一个！")

        # 为未提供的可选参数设置空张量
        if shs is None:
            shs = torch.Tensor([])              # 球谐函数系数
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])   # 预计算颜色

        if scales is None:
            scales = torch.Tensor([])           # 缩放参数
        if rotations is None:
            rotations = torch.Tensor([])        # 旋转四元数
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])    # 预计算3D协方差
        if theta is None:
            theta = torch.Tensor([])            # MCMC theta参数
        if rho is None:
            rho = torch.Tensor([])              # MCMC rho参数

        # 调用C++/CUDA光栅化程序
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings,
        )


def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    """
    计算高斯重定位
    
    在MCMC采样过程中，根据统计信息重新计算高斯的不透明度和缩放参数
    这是MCMC版本特有的功能，用于优化高斯分布
    
    Args:
        opacity_old: 旧的不透明度值
        scale_old: 旧的缩放参数
        N: 采样数量
        binoms: 二项式系数
        n_max: 最大采样数
        
    Returns:
        tuple: (new_opacity, new_scale) 新的不透明度和缩放参数
    """
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale
