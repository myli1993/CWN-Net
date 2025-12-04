class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, num_quarters=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.window_size = window_size
        self.num_quarters = num_quarters

    def forward(self, x):
        b, seq_len, dim = x.shape
        if dim != self.attn.embed_dim or dim != self.temporal_attn.embed_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.attn.embed_dim}, got {dim}")
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        print(f"Before temporal_attn: x.shape = {x.shape}")
        x_temporal = x.view(b, -1, self.num_quarters, dim)
        x_temporal = x_temporal.transpose(1, 2).reshape(b * self.num_quarters, -1, dim)
        print(f"Before temporal_attn reshape: x_temporal.shape = {x_temporal.shape}")
        if torch.isnan(x_temporal).any() or torch.isinf(x_temporal).any():
            print(f"NaN or Inf detected in x_temporal for batch {b}, seq_len {seq_len}")
            x_temporal = torch.nan_to_num(x_temporal, nan=0.0, posinf=0.0, neginf=0.0)
        x_temporal, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = x_temporal.view(b, self.num_quarters, -1, dim).transpose(1, 2).reshape(b, seq_len, dim)
        x = x + x_temporal
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x
        return x