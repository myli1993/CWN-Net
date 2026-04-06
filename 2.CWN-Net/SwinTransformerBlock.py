import torch
import torch.nn as nn

class CorrectedSwinTemporalBlock(nn.Module):
    # 根据论文，建议将默认参数调整为论文一致的值，或在外部传入
    def __init__(self, dim, num_heads=3, window_size=7, num_quarters=4):
        super().__init__()
        self.dim = dim
        self.num_quarters = num_quarters
        self.window_size = window_size # 真正实现Swin时需要用到此参数划分窗口
        
        # 1. 空间注意力层 (注：目前这里依然是全局注意力。如果要变成Swin，需换成WindowAttention)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 2. 时间注意力层
        self.norm_temporal = nn.LayerNorm(dim) # 建议为时间注意力也加上独立的 Norm
        self.temporal_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 3. MLP 层
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x.shape = (B, seq_len, dim)
        b, seq_len, dim = x.shape
        
        if dim != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {dim}")

        # ================================
        # Block 1: 空间注意力 + 残差
        # ================================
        shortcut = x  # 保存残差 base
        x_norm = self.norm1(x)
        # 注意：这里若要真正成为 Swin，需在此处进行 window_partition 切分，算完后再 window_reverse
        x_spatial, _ = self.attn(x_norm, x_norm, x_norm)
        x = shortcut + x_spatial # 正确的残差相加

        # ================================
        # Block 2: 时间注意力 + 残差
        # ================================
        shortcut_temp = x
        x_norm_temp = self.norm_temporal(x)
        
        # 准备时间注意力的输入形状
        # 假设 seq_len 是由于空间展平产生的 (e.g., H*W)。
        # 这里您的原始时间切分逻辑需要确保维度的物理意义正确！
        x_temporal = x_norm_temp.view(b, -1, self.num_quarters, dim)
        x_temporal = x_temporal.transpose(1, 2).reshape(b * self.num_quarters, -1, dim)
        
        # 异常值处理
        if torch.isnan(x_temporal).any() or torch.isinf(x_temporal).any():
            x_temporal = torch.nan_to_num(x_temporal, nan=0.0, posinf=0.0, neginf=0.0)
            
        temporal_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        
        # 恢复形状
        temporal_out = temporal_out.view(b, self.num_quarters, -1, dim).transpose(1, 2).reshape(b, seq_len, dim)
        
        x = shortcut_temp + temporal_out # 正确的残差相加

        # ================================
        # Block 3: MLP + 残差
        # ================================
        shortcut_mlp = x
        x_norm_mlp = self.norm2(x)
        x_mlp = self.mlp(x_norm_mlp)
        x = shortcut_mlp + x_mlp # 修复原代码中荒谬的 x = x + x
        
        return x
