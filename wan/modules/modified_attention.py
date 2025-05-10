import torch
import torch.nn as nn

from .model import rope_apply
from .attention import flash_attention

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
                self.features_dict["query"][self.layer_idx] = q
                self.features_dict["key"][self.layer_idx] = k
        
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

class ModifiedCrossAttention(ModifiedSelfAttention):

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
                self.features_dict["query"][self.layer_idx] = q.detach()
                self.features_dict["key"][self.layer_idx] = k.detach()

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

