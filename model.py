"""
复现：基于动态采样与多尺度特征增强的 DeepLabV3+ 滑坡语义分割（蔡丹丹等，2026）
模型定义。三个改进模块均可通过开关关闭，用于复现消融实验（论文表3）。

注意：论文未给出的细节（采样网格间距、GAM 缩减率、解码器卷积类型等）
已在注释中标明 [ASSUMPTION]，可按需调整。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def cbr(c_in, c_out, k=1, s=1, p=0, d=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, dilation=d, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


# ----------------------------------------------------------------------
# 1. RG-LDConv：规则网格线性可变形卷积（论文 1.1 节，式(1)(2)，图2a）
# ----------------------------------------------------------------------
class RGLDConv(nn.Module):
    """
    流程（对应图2a）：
      3x3 卷积预测 2N 通道偏移 ΔP  ->  P = P0 + Pn + ΔP  ->  边界截断
      -> 双线性重采样 -> 重排 (B,C,H,W,N)->(B,C,H,W*N) -> 1xN 卷积(stride 1xN)+BN+ReLU
    k: 规则网格边长，N = k*k（论文：N=9 即 k=3；N=16 即 k=4）
    grid_step: 规则网格点间距（以特征图像素为单位）。
      [ASSUMPTION] 论文未给出该值，这里类比 ASPP 的膨胀率设置，作为首要调参项。
    """

    def __init__(self, c_in, c_out, k=3, grid_step=1.0):
        super().__init__()
        self.k, self.N = k, k * k
        self.offset_conv = nn.Conv2d(c_in, 2 * self.N, 3, padding=1)
        # 零初始化：训练初期等价于规则网格采样（DCN/LDConv 常用做法）
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        r = (torch.arange(k, dtype=torch.float32) - (k - 1) / 2) * grid_step
        gy, gx = torch.meshgrid(r, r, indexing="ij")
        pn = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=0)  # (2, N)
        self.register_buffer("pn", pn.view(1, 2, self.N, 1, 1), persistent=False)

        self.proj = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(1, self.N), stride=(1, self.N), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.N
        off = self.offset_conv(x).float().view(B, 2, N, H, W)  # 前N通道为x偏移，后N为y偏移

        ys, xs = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32),
            torch.arange(W, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        p0 = torch.stack([xs, ys], dim=0).view(1, 2, 1, H, W)
        p = p0 + self.pn + off  # 式(1)  (B,2,N,H,W)

        # 归一化到 [-1,1]；padding_mode="border" 即论文中的 Boundary Clipping
        gx = 2.0 * p[:, 0] / max(W - 1, 1) - 1.0
        gy = 2.0 * p[:, 1] / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(B, N * H, W, 2)

        # 式(2)：对4个整数邻点的双线性加权
        samp = F.grid_sample(x, grid.to(x.dtype), mode="bilinear",
                             padding_mode="border", align_corners=True)  # (B,C,N*H,W)
        samp = samp.view(B, C, N, H, W).permute(0, 1, 3, 4, 2).reshape(B, C, H, W * N)
        return self.proj(samp)


class RGLDConvASPP(nn.Module):
    """图2b：1x1 分支 + 3个 RG-LDConv 分支(N=9,9,16) + 全局池化分支 -> concat -> 1x1 融合(+Dropout 0.1)"""

    def __init__(self, c_in=320, c=256, cfgs=((3, 6.0), (3, 6.0), (4, 12.0))):
        # ————建议修改的采样网格的间距，间距6、12偏大。可试 (3,1),(3,1),(4,2) 和 (3,2),(3,2),(4,4)
        # cfgs: (k, grid_step)。前两个分支"相同基础采样尺度、独立参数"，第三个扩大覆盖范围 —— [ASSUMPTION]
        super().__init__()
        self.b0 = cbr(c_in, c, 1)
        self.branches = nn.ModuleList([RGLDConv(c_in, c, k, s) for k, s in cfgs])
        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(c_in, c, 1, bias=False),
                                nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(c * (len(cfgs) + 2), c, 1, bias=False),
                                  nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        outs = [self.b0(x)] + [b(x) for b in self.branches]
        g = F.interpolate(self.gp(x), size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat(outs + [g], dim=1))


class PlainASPP(nn.Module):
    """baseline：标准 ASPP（膨胀率 6/12/18）"""

    def __init__(self, c_in=320, c=256, rates=(6, 12, 18)):
        super().__init__()
        self.b0 = cbr(c_in, c, 1)
        self.branches = nn.ModuleList([cbr(c_in, c, 3, 1, r, r) for r in rates])
        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(c_in, c, 1, bias=False),
                                nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(c * (len(rates) + 2), c, 1, bias=False),
                                  nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        outs = [self.b0(x)] + [b(x) for b in self.branches]
        g = F.interpolate(self.gp(x), size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat(outs + [g], dim=1))


# ----------------------------------------------------------------------
# 2. GAM-GFE：通道注意力 -> 空间注意力（论文 1.2 节，图3），置于 ASPP 之后
# ----------------------------------------------------------------------
class GAMGFE(nn.Module):
    def __init__(self, c=256, r=4):  # [ASSUMPTION] r=4 为 GAM 原文默认值
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c))
        self.spatial = nn.Sequential(
            nn.Conv2d(c, c // r, 7, padding=3), nn.BatchNorm2d(c // r), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 7, padding=3), nn.BatchNorm2d(c),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        a = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        a = self.mlp(a).view(B, H, W, C).permute(0, 3, 1, 2)
        xc = x * torch.sigmoid(a)                    # 通道细化特征 Xc
        return xc * torch.sigmoid(self.spatial(xc))  # 空间聚焦


# ----------------------------------------------------------------------
# 3. MRFF：多尺度残差特征融合（论文 1.3 节，式(3)-(9)，图4）
# ----------------------------------------------------------------------
class MRFF(nn.Module):
    def __init__(self, c_in=304, c=256):
        super().__init__()
        self.align = cbr(c_in, c, 1)                                   # 式(4)
        self.b1 = cbr(c, 86, 1)                                        # 式(5) 原分辨率
        self.b2 = cbr(c, 85, 3, 2, 1)                                  # 式(6) 步长2的3x3卷积
        self.b3 = nn.Sequential(nn.MaxPool2d(3, 2, 1), cbr(c, 85, 1))  # 式(7) 池化分支
        self.fuse = cbr(256, 256, 3, 1, 1)                             # 式(8)

    def forward(self, x):
        fa = self.align(x)
        size = fa.shape[2:]
        f1 = self.b1(fa)
        f2 = F.interpolate(self.b2(fa), size=size, mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.b3(fa), size=size, mode="bilinear", align_corners=False)
        fm = self.fuse(torch.cat([f1, f2, f3], dim=1))
        return fm + fa                                                 # 式(9) 残差


# ----------------------------------------------------------------------
# 整体网络（图1）
# ----------------------------------------------------------------------
class LandslideDeepLab(nn.Module):
    def __init__(self, num_classes=2, pretrained=True,
                 use_ldaspp=True, use_gam=True, use_mrff=True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        feats = mobilenet_v2(weights=weights).features
        # 输出步幅16：block14 的 stride 2 -> 1；block15-17 使用 dilation 2
        feats[14].conv[1][0].stride = (1, 1)
        for i in (15, 16, 17):
            dw = feats[i].conv[1][0]
            dw.dilation, dw.padding = (2, 2), (2, 2)
        self.low = feats[:4]       # Stage1: 1/4, C=24
        self.high = feats[4:18]    # Stage2-5: 1/16, C=320

        self.aspp = RGLDConvASPP() if use_ldaspp else PlainASPP()
        self.gam = GAMGFE() if use_gam else nn.Identity()
        self.low_proj = cbr(24, 48, 1)
        self.mrff = MRFF(304, 256) if use_mrff else nn.Identity()
        dec_in = 256 if use_mrff else 304
        # [ASSUMPTION] 图1 写作 "3x3 Conv Block x2"，此处用普通 3x3 Conv-BN-ReLU（非深度可分离）
        self.dec = nn.Sequential(cbr(dec_in, 256, 3, 1, 1), cbr(256, 256, 3, 1, 1))
        self.cls = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        low = self.low(x)
        high = self.aspp(self.high(low))
        high = self.gam(high)
        high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)  # 4x
        z = torch.cat([high, self.low_proj(low)], dim=1)  # 256+48=304
        z = self.mrff(z)
        z = self.cls(self.dec(z))
        return F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    m = LandslideDeepLab(pretrained=False)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    y = m(torch.randn(2, 3, 512, 512))
    print("out:", tuple(y.shape), f"params: {n:.2f}M  (论文: 9.02M)")
    base = LandslideDeepLab(pretrained=False, use_ldaspp=False, use_gam=False, use_mrff=False)
    nb = sum(p.numel() for p in base.parameters()) / 1e6
    print(f"baseline params: {nb:.2f}M  (论文: 5.81M)")
