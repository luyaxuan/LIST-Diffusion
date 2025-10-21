import torch
import torch.nn as nn

class TACRPlus(nn.Module):
    def __init__(self, cond_dim, time_dim, out_dim):
        super().__init__()
        # 条件投影（两路）
        self.global_proj = nn.Linear(cond_dim, out_dim)
        self.local_proj = nn.Linear(cond_dim, out_dim)

        # 时间步嵌入建模：Conv1D + MLP
        self.time_conv_mlp = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),   # 输入 shape: [B, 1, T]
            nn.ReLU(),
            nn.Flatten(),                                # 展平为 [B, 8 * T]
            nn.Linear(8 * time_dim, 2 * out_dim),        # 输出 λ_g 和 λ_l
            nn.Softmax(dim=-1)                           # 归一化为 gating 权重
        )

        # 可选：残差调制的输出卷积（可以不加）
        self.out_proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, h, cond_vec, t_embed):
        """
        h        : [B, C, H, W] - UNet 中间 feature
        cond_vec : [B, cond_dim] - 文本 + 传感器编码后拼接向量
        t_embed  : [B, time_dim] - 扩散时间步嵌入向量
        """
        B, C, H, W = h.shape

        # 条件编码两个通道
        c_global = self.global_proj(cond_vec)  # [B, C]
        c_local = self.local_proj(cond_vec)    # [B, C]

        # 时间步嵌入通过 1D 卷积建模后得到 gating 权重
        t_conv_input = t_embed.unsqueeze(1)  # [B, 1, T]
        lambda_all = self.time_conv_mlp(t_conv_input)  # [B, 2 * C]
        lambda_g, lambda_l = lambda_all.chunk(2, dim=-1)  # [B, C], [B, C]

        # 加权组合两路条件
        delta = lambda_g * c_global + lambda_l * c_local  # [B, C]
        delta = delta.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, C, H, W]

        # 残差融合
        return h + self.out_proj(delta) 