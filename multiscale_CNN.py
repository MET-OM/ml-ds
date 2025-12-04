import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# Multiscale convolution block
# ----------------------------------------
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7], dilations=[1,2,3],
                 use_batchnorm=False, dropout=0.0):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            for d in dilations:
                padding = ((k - 1) // 2) * d  # maintain same spatial size
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=padding, dilation=d)
                layers = [conv]
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0.0:
                    layers.append(nn.Dropout2d(dropout))
                self.branches.append(nn.Sequential(*layers))
        self.fuse = nn.Conv2d(out_channels * len(self.branches), out_channels, kernel_size=1)

    def forward(self, x):
        out = [branch(x) for branch in self.branches]
        x_cat = torch.cat(out, dim=1)
        return F.relu(self.fuse(x_cat))


# ----------------------------------------
# Standard residual block
# ----------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels, use_batchnorm=False, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(channels, channels, 3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, channels, 3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(x + self.conv(x))


# ----------------------------------------
# Full ConvResNet with multiscale frontend
# ----------------------------------------
class MultiScaleConvResNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, num_resblocks=4,
                 kernel_sizes=[3,5,7], dilations=[1,2,3], use_batchnorm=False, dropout=0.0):
        super().__init__()
        self.multiscale = MultiScaleConv(
            in_channels, base_channels, kernel_sizes, dilations,
            use_batchnorm=use_batchnorm, dropout=dropout
        )
        self.resblocks = nn.Sequential(
            *[ResBlock(base_channels, use_batchnorm=use_batchnorm, dropout=dropout)
              for _ in range(num_resblocks)]
        )
        self.head = nn.Conv2d(base_channels, out_channels, 3, padding=1)  # output same resolution

    def forward(self, x):
        x = self.multiscale(x)
        x = self.resblocks(x)
        x = self.head(x)
        return x


# ----------------------------------------
# Example
# ----------------------------------------
if __name__ == "__main__":
    model = MultiScaleConvResNet(
        in_channels=10, out_channels=1, base_channels=64, num_resblocks=6,
        kernel_sizes=[3,5], dilations=[1,2,3],
        use_batchnorm=True, dropout=0.1
    )
    x = torch.randn(2, 10, 32, 32)  # batch, channels, height, width
    y = model(x)
    print(y.shape)  # -> (2, 1, 32, 32)
