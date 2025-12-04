class UNetCnnSwinImage(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            ChannelAttention(64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            ChannelAttention(128)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.patch_embed = nn.Conv2d(256, 256, kernel_size=4, stride=4)
        self.swin1 = SwinTransformerBlock(dim=256, num_heads=8, window_size=8)
        self.swin2 = SwinTransformerBlock(dim=256, num_heads=8, window_size=8)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        print(f"e1 shape: {e1.shape}")
        e2 = self.enc2(e1)
        print(f"e2 shape: {e2.shape}")
        e3 = self.enc3(e2)
        print(f"e3 shape: {e3.shape}")
        b = self.patch_embed(e3)
        print(f"b shape after patch_embed: {b.shape}")
        b = b.flatten(2).transpose(1, 2)
        print(f"b shape after flatten/transpose: {b.shape}")
        b = self.swin1(b)
        b = self.swin2(b)
        b = b.transpose(1, 2).view(b.size(0), 256, 8, 8)
        print(f"b shape after swin: {b.shape}")
        d3 = self.upconv3(b)
        print(f"d3 shape: {d3.shape}, d3 NaN/Inf: {torch.isnan(d3).any() or torch.isinf(d3).any()}")
        e2_resized = F.interpolate(e2, size=(d3.size(2), d3.size(3)), mode='bilinear', align_corners=False)
        print(f"e2_resized shape: {e2_resized.shape}")
        if d3.size(1) + e2_resized.size(1) != 256:
            raise ValueError(f"Channel mismatch: d3 ({d3.size(1)}) + e2_resized ({e2_resized.size(1)}) != 256")
        d3 = torch.cat([d3, e2_resized], dim=1)
        if torch.isnan(d3).any() or torch.isinf(d3).any():
            print(f"NaN or Inf detected in d3, replacing with 0")
            d3 = torch.nan_to_num(d3, nan=0.0, posinf=0.0, neginf=0.0)
        d3 = self.dec3(d3)
        print(f"d3 after dec3 shape: {d3.shape}, d3 NaN/Inf: {torch.isnan(d3).any() or torch.isinf(d3).any()}")
        d2 = self.upconv2(d3)
        print(f"d2 shape: {d2.shape}, d2 NaN/Inf: {torch.isnan(d2).any() or torch.isinf(d2).any()}")
        e1_resized = F.interpolate(e1, size=(d2.size(2), d2.size(3)), mode='bilinear', align_corners=False)
        print(f"e1_resized shape: {e1_resized.shape}")
        if d2.size(1) + e1_resized.size(1) != 128:
            raise ValueError(f"Channel mismatch: d2 ({d2.size(1)}) + e1_resized ({e1_resized.size(1)}) != 128")
        d2 = torch.cat([d2, e1_resized], dim=1)
        if torch.isnan(d2).any() or torch.isinf(d2).any():
            print(f"NaN or Inf detected in d2, replacing with 0 at {torch.where(torch.isnan(d2) | torch.isinf(d2))}")
            d2 = torch.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0)
        d2 = self.dec2(d2)
        print(f"d2 after dec2 shape: {d2.shape}, d2 NaN/Inf: {torch.isnan(d2).any() or torch.isinf(d2).any()}")
        out = self.out(d2)
        print(f"out shape: {out.shape}, out NaN/Inf: {torch.isnan(out).any() or torch.isinf(out).any()}")
        out = self.sigmoid(out)
        return out
