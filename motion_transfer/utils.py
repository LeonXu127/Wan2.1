import decord
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math
from einops import rearrange

def video_to_tensor(video_path, num_frames, target_size, device):
    """处理视频文件，返回指定帧数的视频张量
    
    Args:
        video_path (str): 视频文件路径
        num_frames (int): 需要采样的帧数
        target_size (tuple): 目标尺寸(宽度,高度)
        device (torch.device): 设备
        
    Returns:
        torch.Tensor: 处理后的视频张量，形状为 [C, F, H, W]
    """
    logging.info(f"读取视频文件: {video_path}")
    try:
        # 使用 decord 读取视频
        vr = decord.VideoReader(video_path)
        if num_frames:
            frame_indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
            logging.info(f"选择视频帧索引: {frame_indices}")
        else:
            frame_indices = np.arange(len(vr))

        # 读取指定帧
        frames = torch.Tensor(vr.get_batch(frame_indices).asnumpy())  # Shape: [T, H, W, C]
        frames = rearrange(frames, "f h w c -> f c w h")
        frames = torch.nn.functional.interpolate(frames, size=target_size, mode="bilinear", align_corners=True)
        video_tensor = rearrange(frames, "f c w h -> c f h w")
        video_tensor = video_tensor / 127.5 - 1.0
        
        # 将视频张量移到设备上
        video_tensor = video_tensor.to(device)
        logging.info(f"处理后的视频张量形状: {video_tensor.shape}")
        return video_tensor
        
    except Exception as e:
        logging.error(f"读取视频文件失败: {e}")
        raise e

def pos_1dto2d(dest, W):
    h = dest // W
    w = dest % W
    
    return h, w

def cal_flowij(
    Aij: torch.Tensor,
    reference: bool,
    N: int,
    H: int,
    W: int,
):
    S = H * W
    if reference:
        with torch.no_grad():
            dest = Aij.argmax(dim=1)
            flow_ij = torch.stack(pos_1dto2d(torch.arange(S, device=Aij.device), W), dim=-1) \
                                - torch.stack(pos_1dto2d(dest, W), dim=-1)
    else:
        pos = torch.meshgrid(
            torch.arange(H, device=Aij.device),
            torch.arange(W, device=Aij.device),
            indexing="ij",
        )
        flow = [pos[i].flatten().unsqueeze(0) - pos[i].flatten().unsqueeze(1) for i in range(len(pos))]
        flow_ij = torch.stack([torch.sum(flow[i] * Aij, dim=1) for i in range(len(flow))], dim=-1)
    
    return flow_ij

def cal_pooling_flowij(
    Aij: torch.Tensor,
    reference: bool,
    N: int,
    H: int,
    W: int,
    i: int,
    j: int,
    pooling_size: tuple,
):
    S = H * W
    if reference:
        with torch.no_grad():
            # 可视化Aij相似度分布
            # prob = Aij.max(dim=1)[0]
            # bins = torch.linspace(0, 0.2, 20)
            # import matplotlib.pyplot as plt
            # plt.clf()
            # counts, bins, patches = plt.hist(prob.cpu().numpy(), bins=bins, edgecolor='black')
            # for k in range(len(counts)):
            #     bin_center = (bins[k] + bins[k+1]) / 2
            #     plt.text(bin_center, counts[k], f"{counts[k]:.0f}", ha='center', va='bottom')
            # plt.title(f"similarity between frame {i} and frame {j}")
            # plt.savefig(f"motion_transfer/logs/prob.png")
            # input(f"check prob_{i}_{j}")
            dest = Aij.argmax(dim=1)
            flow_ij = torch.stack(pos_1dto2d(torch.arange(S, device=Aij.device), W), dim=-1) \
                                - torch.stack(pos_1dto2d(dest, W), dim=-1)
    else:
        pos = torch.meshgrid(
            torch.arange(H, device=Aij.device),
            torch.arange(W, device=Aij.device),
            indexing="ij",
        )
        flow = [pos[i].flatten().unsqueeze(0) - pos[i].flatten().unsqueeze(1) for i in range(len(pos))]
        flow_ij = torch.stack([torch.sum(flow[i] * Aij, dim=1) for i in range(len(flow))], dim=-1)
    
    return flow_ij

def cal_motion_flow(
    query: torch.Tensor,
    key: torch.Tensor,
    latent_shape: tuple,
    reference: bool = False
):
    nheads = query.shape[2]
    N = latent_shape[0]
    H = latent_shape[1]
    W = latent_shape[2]
    S = H * W
    # print(f"latent shape nheads:{nheads}, N:{N}, H:{H}, W:{W}, S:{S}")
    # input("check N, H, W, S")

    motion_flow = []

    if not reference:
        query = query.cpu()
        key = key.cpu()

    for i in range(nheads):
        q = query[:, :, i, :].squeeze(0)
        k = key[:, :, i, :].squeeze(0)
        # if not reference:
        #     input("check gpu memory")
        A_head = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        A_head = rearrange(A_head, "(n1 s1) (n2 s2) -> n1 n2 s1 s2", n1=N, n2=N, s1=S, s2=S)

        for i in range(N):
            for j in range(N):
                Aij = A_head[i, j, :, :]

                flow_ij = cal_flowij(Aij, reference, N, H, W)
                
                motion_flow.append(flow_ij) 

                del Aij, flow_ij
                torch.cuda.empty_cache()
        del A_head
        torch.cuda.empty_cache()

    motion_flow = torch.stack(motion_flow, dim=0).view(nheads, N, N, H, W, 2).to(dtype=query.dtype)

    return motion_flow if not reference else motion_flow.cpu()

def cal_motion_flow_selectedhead(
    query: torch.Tensor,
    key: torch.Tensor,
    latent_shape: tuple,
    reference: bool = False
):
    nheads = query.shape[2]
    N = latent_shape[0]
    H = latent_shape[1]
    W = latent_shape[2]
    S = H * W
    print(f"latent shape nheads:{nheads}, N:{N}, H:{H}, W:{W}, S:{S}")

    motion_flow = []

    # 取2个head
    query = query[:, :, :2, :].squeeze(0)
    key = key[:, :, :2, :].squeeze(0)
    num_heads = query.shape[1]
    for i in range(num_heads):
        q = query[ :, i, :].squeeze(0)
        k = key[ :, i, :].squeeze(0)
        A_head = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        A_head = rearrange(A_head, "(n1 s1) (n2 s2) -> n1 n2 s1 s2", n1=N, n2=N, s1=S, s2=S)

        for i in range(N):
            for j in range(N):
                Aij = A_head[i, j, :, :]
                
                flow_ij = cal_flowij(Aij, reference, N, H, W)
                # flow_ij = cal_pooling_flowij(Aij, reference, N, H, W, i, j, (2, 2))
                motion_flow.append(flow_ij)

                del Aij, flow_ij
                torch.cuda.empty_cache()
        del A_head
        torch.cuda.empty_cache()

    motion_flow = torch.stack(motion_flow, dim=0).view(num_heads, N, N, H, W, 2).to(dtype=query.dtype)

    return motion_flow if not reference else motion_flow.cpu()

# TODO: 看一下生成的这俩函数
def max_pooling_with_indices(attention_map, kernel_size=(2, 2), stride=None):
    """对注意力图进行最大池化，并返回每个池化窗口中最大值的原始坐标
    
    Args:
        attention_map (torch.Tensor): 输入的注意力图，形状为 [H, W] 或 [B, H, W]
        kernel_size (tuple): 池化窗口大小，默认 (2, 2)
        stride (tuple): 步长，默认与kernel_size相同
        
    Returns:
        tuple: (pooled_map, indices)
            pooled_map: 池化后的注意力图
            indices: 每个池化窗口中最大值在原始注意力图中的坐标 (h, w)
    """
    if stride is None:
        stride = kernel_size
        
    # 确保输入是3D的 [B, H, W]
    if len(attention_map.shape) == 2:
        attention_map = attention_map.unsqueeze(0)
        
    B, H, W = attention_map.shape
    kh, kw = kernel_size
    sh, sw = stride
    
    # 计算输出特征图大小
    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1
    
    # 初始化输出和索引
    pooled_map = torch.zeros((B, out_h, out_w), device=attention_map.device)
    indices = torch.zeros((B, out_h, out_w, 2), device=attention_map.device, dtype=torch.long)
    
    # 执行池化并记录索引
    for b in range(B):
        for i in range(out_h):
            for j in range(out_w):
                # 当前窗口
                h_start, w_start = i * sh, j * sw
                h_end, w_end = h_start + kh, w_start + kw
                window = attention_map[b, h_start:h_end, w_start:w_end]
                
                # 找到最大值及其索引
                max_val, flat_idx = torch.max(window.view(-1), dim=0)
                # 将一维索引转回二维坐标
                local_h, local_w = flat_idx // kw, flat_idx % kw
                
                # 记录全局坐标
                global_h, global_w = h_start + local_h, w_start + local_w
                
                # 保存结果
                pooled_map[b, i, j] = max_val
                indices[b, i, j, 0] = global_h
                indices[b, i, j, 1] = global_w
    
    # 如果原始输入是2D的，则去掉批次维度
    if attention_map.shape[0] == 1 and len(attention_map.shape) == 3:
        pooled_map = pooled_map.squeeze(0)
        indices = indices.squeeze(0)
        
    return pooled_map, indices

def max_pooling_attention_map(attention_map, kernel_size=(2, 2), stride=None):
    """对注意力图进行最大池化，保留原始形状并将非最大值位置置零
    
    Args:
        attention_map (torch.Tensor): 输入的注意力图，形状为 [H, W] 或 [B, H, W]
        kernel_size (tuple): 池化窗口大小，默认 (2, 2)
        stride (tuple): 步长，默认与kernel_size相同
        
    Returns:
        torch.Tensor: 池化后的注意力图，形状与输入相同，但非最大值位置置零
    """
    if stride is None:
        stride = kernel_size
        
    # 保存原始形状信息
    original_shape = attention_map.shape
    original_dim = len(original_shape)
    
    # 确保输入是3D的 [B, H, W]
    if original_dim == 2:
        attention_map = attention_map.unsqueeze(0)
        
    B, H, W = attention_map.shape
    
    # 获取池化结果和索引
    _, indices = max_pooling_with_indices(attention_map, kernel_size, stride)
    
    # 创建与原始attention_map相同形状的零张量
    sparse_map = torch.zeros_like(attention_map)
    
    # 将最大值位置的值设为1
    for b in range(B):
        for i in range(indices.shape[1]):
            for j in range(indices.shape[2]):
                h, w = indices[b, i, j, 0], indices[b, i, j, 1]
                sparse_map[b, h, w] = attention_map[b, h, w]
    
    # 恢复原始维度
    if original_dim == 2:
        sparse_map = sparse_map.squeeze(0)
        
    return sparse_map
