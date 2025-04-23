import decord
import torch
import numpy as np
import logging
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