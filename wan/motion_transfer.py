# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
import numpy as np
import decord
from einops import rearrange

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import torch.nn as nn

from .distributed.fsdp import shard_model
from .modules.model import WanModel, rope_apply
from .modules.attention import flash_attention
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from motion_transfer.utils import video_to_tensor


class ModifiedSelfAttention(nn.Module):
    """修改后的注意力模块，用于提取query和key特征"""
    
    def __init__(self, original_attn, layer_idx):
        super().__init__()
        self.original_attn = original_attn
        self.layer_idx = layer_idx
        self.save_features = False
        
        # 直接从原始注意力模块获取属性
        self.dim = original_attn.dim
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.window_size = original_attn.window_size
        self.qk_norm = original_attn.qk_norm
        self.eps = original_attn.eps
        
        # 直接使用原始注意力模块的层
        self.q = original_attn.q
        self.k = original_attn.k
        self.v = original_attn.v
        self.o = original_attn.o
        self.norm_q = original_attn.norm_q
        self.norm_k = original_attn.norm_k
        
        # 存储特征的字典
        self.features_dict = None
    
    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        处理自注意力并提取特征
        
        Args:
            x (Tensor): 形状 [B, L, C]
            seq_lens (Tensor): 形状 [B]
            grid_sizes (Tensor): 形状 [B, 3]
            freqs (Tensor): Rope频率，形状 [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # 计算query, key, value
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        
        # 在启用特征保存时保存query和key
        if self.save_features:
            if self.features_dict is not None:
                self.features_dict["query"][self.layer_idx] = q.detach().cpu()
                self.features_dict["key"][self.layer_idx] = k.detach().cpu()
        
        # 应用rope并计算注意力
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # 输出
        x = x.flatten(2)
        x = self.o(x)
        return x
    
    def enable_save_features(self, features_dict):
        """启用特征保存"""
        self.save_features = True
        self.features_dict = features_dict
        
    def disable_save_features(self):
        """禁用特征保存"""
        self.save_features = False

class ModifiedT2VCrossAttention(ModifiedSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # 在启用特征保存时保存query和key
        if self.save_features:
            if self.features_dict is not None:
                self.features_dict["query"][self.layer_idx] = q.detach().cpu()
                self.features_dict["key"][self.layer_idx] = k.detach().cpu()

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanMT:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        video_name=None,
    ):
        """
        初始化WanMT模型组件。

        Args:
            config (EasyDict):
                包含从config.py初始化的模型参数的对象
            checkpoint_dir (`str`):
                包含模型检查点的目录路径
            device_id (`int`,  *optional*, defaults to 0):
                目标GPU设备ID
            rank (`int`,  *optional*, defaults to 0):
                分布式训练的进程排名
            t5_fsdp (`bool`, *optional*, defaults to False):
                为T5模型启用FSDP分片
            dit_fsdp (`bool`, *optional*, defaults to False):
                为DiT模型启用FSDP分片
            use_usp (`bool`, *optional*, defaults to False):
                启用USP的分发策略。
            t5_cpu (`bool`, *optional*, defaults to False):
                是否将T5模型放在CPU上。仅在不使用t5_fsdp时有效。
            video_name (`str`, *optional*, defaults to None):
                视频名称，用于保存特征文件时的文件名
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.extract_timestep = config.extract_timestep

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        
        # 创建保存特征的文件夹
        self.save_path = "motion_transfer/ref_ft"
        os.makedirs(self.save_path, exist_ok=True)
        
        # 设置特征文件名
        video_name = os.path.splitext(os.path.basename(video_name))[0]
        self.feature_filename = f"{video_name}.pt"
        
        # 收集的特征
        self.ref_ft = {}
    
    def _modify_attention_modules(self):
        """修改模型的注意力模块，以便在扩散过程中提取特征"""
        # 为所有层创建特征字典
        num_layers = len(self.model.blocks)
        self.ref_ft = {
            "query": {i: None for i in range(num_layers)},
            "key": {i: None for i in range(num_layers)}
        }
        
        for i, block in enumerate(self.model.blocks):
            # # 保存原始self_attn模块的reference
            # original_attn = block.self_attn
            # # 创建修改后的注意力模块
            # modified_attn = ModifiedSelfAttention(
            #     original_attn=original_attn,
            #     layer_idx=i
            # )
            # # 替换原始注意力模块
            # block.self_attn = modified_attn
            # 保存原始cross_attn模块的reference
            original_attn = block.cross_attn
            # 创建修改后的注意力模块
            modified_attn = ModifiedT2VCrossAttention(
                original_attn=original_attn,
                layer_idx=i
            )
            # 替换原始注意力模块
            block.cross_attn = modified_attn
            
    def _restore_attention_modules(self):
        """恢复原始的注意力模块"""
        for block in self.model.blocks:
            # if isinstance(block.self_attn, ModifiedSelfAttention):
            #     block.self_attn = block.self_attn.original_attn
            if isinstance(block.cross_attn, ModifiedT2VCrossAttention):
                block.cross_attn = block.cross_attn.original_attn
    
    def load_features(self, feature_path=None):
        """
        加载保存的特征
        
        Args:
            feature_path (str, optional): 
                特征文件的路径，如果为None，使用默认路径
        
        Returns:
            dict: 包含query和key的字典
        """
        self.ref_ft = torch.load(feature_path)
        print(f"feature_shape: {self.ref_ft['query'][0].shape}")
        return self.ref_ft
    
    def save_features(self, feature_path=None):
        """
        保存提取的特征
        
        Args:
            feature_path (str, optional): 
                保存特征的路径，如果为None，使用默认路径
        """
        if feature_path is None:
            feature_path = os.path.join(self.save_path, self.feature_filename)
        print(f"feature_shape: {self.ref_ft['query'][0].shape}")
        torch.save(self.ref_ft, feature_path)
        logging.info(f"特征已保存到 {feature_path}")

    def extract_features(self,
                 input_prompt,
                 input_video_path,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        """
        从输入的提示和视频中提取特征，在diffusion过程中的特定timestep保存query和key。

        Args:
            input_prompt (`str`):
                文本提示
            input_video_path (`str`):
                输入视频文件路径
            size (tuple[`int`], *optional*, defaults to (1280,720)):
                控制视频分辨率，(宽度,高度)
            frame_num (`int`, *optional*, defaults to 81):
                要从视频中采样的帧数。数字应该是4n+1
            shift (`float`, *optional*, defaults to 5.0):
                噪声调度偏移参数。影响时间动态
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                用于采样视频的求解器
            sampling_steps (`int`, *optional*, defaults to 50):
                扩散采样步骤数
            guide_scale (`float`, *optional*, defaults 5.0):
                无分类器引导比例
            n_prompt (`str`, *optional*, defaults to ""):
                负面提示。如果未给出，使用`config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                噪声生成的随机种子。如果为-1，使用随机种子
            offload_model (`bool`, *optional*, defaults to True):
                如果为True，在生成过程中将模型卸载到CPU以节省VRAM
        """
        # 检查特征文件是否已存在
        feature_path = os.path.join(self.save_path, self.feature_filename)
        if os.path.exists(feature_path):
            logging.info(f"特征文件 {feature_path} 已存在，直接加载")
            self.load_features(feature_path)
            return
            
        # 处理视频文件，返回指定帧数的视频张量
        input_video = video_to_tensor(input_video_path, frame_num, size, self.device)
            
        # 预处理
        if input_video.dim() == 3:  # (C, H, W) -> (C, 1, H, W)
            input_video = input_video.unsqueeze(1)
        
        # 获取视频的尺寸
        C, F, H, W = input_video.shape
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                          H // self.vae_stride[1],
                          W // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                             (self.patch_size[1] * self.patch_size[2]) *
                             target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        
        # 获取输入视频的latent表示
        with torch.no_grad():
            input_latent = self.vae.encode(input_video.unsqueeze(0))[0]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # 修改注意力模块以提取特征
        self._modify_attention_modules()

        # 评估模式
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # 设置输入latent为初始latent
            latent = input_latent
            
            # 仅对目标timestep进行处理
            target_t = self.extract_timestep
            logging.info(f"仅提取timestep={target_t}的特征")
            
            timestep = torch.tensor([target_t], device=self.device)
            
            # 启用特征保存
            for block in self.model.blocks:
                # if isinstance(block.self_attn, ModifiedSelfAttention):
                #     block.self_attn.enable_save_features(self.ref_ft)
                if isinstance(block.cross_attn, ModifiedT2VCrossAttention):
                    block.cross_attn.enable_save_features(self.ref_ft)
            
            # 带条件前向传播，提取特征
            self.model.to(self.device)
            arg_c = {'context': context, 'seq_len': seq_len}
            _ = self.model(
                [latent], t=timestep, **arg_c)[0]
            
            # 禁用特征保存
            for block in self.model.blocks:
                # if isinstance(block.self_attn, ModifiedSelfAttention):
                #     block.self_attn.disable_save_features()
                if isinstance(block.cross_attn, ModifiedT2VCrossAttention):
                    block.cross_attn.disable_save_features()

        # 恢复原始注意力模块
        self._restore_attention_modules()
        
        # 保存提取的特征
        self.save_features()
        
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 ref_ft_path=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            ref_ft_path (`str`, *optional*, defaults to None):
                Path to the reference feature file.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
