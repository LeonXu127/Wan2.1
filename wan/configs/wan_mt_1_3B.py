# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan MT 1.3B ------------------------#

mt_1_3B = EasyDict(__name__='Config: Wan MT 1.3B')
mt_1_3B.update(wan_shared_cfg)

# t5
mt_1_3B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
mt_1_3B.t5_tokenizer = 'google/umt5-xxl'

# vae
mt_1_3B.vae_checkpoint = 'Wan2.1_VAE.pth'
mt_1_3B.vae_stride = (4, 8, 8)

# transformer
mt_1_3B.patch_size = (1, 2, 2)
mt_1_3B.dim = 1536
mt_1_3B.ffn_dim = 8960
mt_1_3B.freq_dim = 256
mt_1_3B.num_heads = 12
mt_1_3B.num_layers = 30
mt_1_3B.window_size = (-1, -1)
mt_1_3B.qk_norm = True
mt_1_3B.cross_attn_norm = True
mt_1_3B.eps = 1e-6

# motion transfer specific
mt_1_3B.extract_timestep = 0  # 提取特定timestep的特征
mt_1_3B.extract_layer_id = 15  # 提取特定layer的特征
mt_1_3B.optimize_steps = 10  # 优化latents步数
mt_1_3B.optimize_timestep_ratio = 0.2 # 优化latents的timestep比例   
mt_1_3B.optimize_lr = 0.005  # 优化latents学习率
