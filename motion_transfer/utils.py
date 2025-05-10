import decord
import torch
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
    # input("check N, H, W, S")

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
                motion_flow.append(flow_ij)

                del Aij, flow_ij
                torch.cuda.empty_cache()
        del A_head
        torch.cuda.empty_cache()

    motion_flow = torch.stack(motion_flow, dim=0).view(num_heads, N, N, H, W, 2).to(dtype=query.dtype)

    return motion_flow if not reference else motion_flow.cpu()
