import torch.nn as nn


def _build_normalization(name: str | None, channels: int) -> nn.Module | None:
    if name is None:
        return None
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "layer":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported normalization '{name}'.")


def _build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")


def _build_dropout(kind: str, rate: float) -> nn.Module | None:
    if rate <= 0:
        return None
    if kind == "spatial":
        return nn.Dropout2d(rate)
    if kind == "standard":
        return nn.Dropout(rate)
    raise ValueError(f"Unsupported dropout variant '{kind}'.")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        normalization=None,
        activation="relu",
        dropout_rate=0.0,
        dropout_variant="spatial",
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = _build_normalization(normalization, out_channels)
        self.activation = _build_activation(activation)
        self.dropout = _build_dropout(dropout_variant, dropout_rate)

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        normalization=None,
        activation="relu",
        dropout_rate=0.0,
        dropout_variant="spatial",
    ):
        super().__init__()
        self.conv1 = ConvBlock(
            channels,
            channels,
            normalization=normalization,
            activation=activation,
            dropout_rate=dropout_rate,
            dropout_variant=dropout_variant,
        )
        self.conv2 = ConvBlock(
            channels,
            channels,
            normalization=normalization,
            activation=activation,
            dropout_rate=dropout_rate,
            dropout_variant=dropout_variant,
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))  # simple residual connection


class ConvResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_filters=8,
        n_blocks=8,
        normalization=None,
        dropout_rate=0.0,
        dropout_variant="spatial",
        attention=False,
        activation="relu",
        localcon_layer=True,
    ):
        super().__init__()

        self.localcon_layer = None
        if localcon_layer:
            self.localcon_layer = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
            in_channels = n_filters

        self.initial_conv = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                ResBlock(n_filters, normalization, activation, dropout_rate, dropout_variant)
                for _ in range(n_blocks)
            ]
        )

        self.attention = None
        if attention:
            raise NotImplementedError("Attention is not implemented in ConvResNet.")

        self.final_conv = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        if self.localcon_layer is not None:
            x = self.localcon_layer(x)
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.final_conv(x)
        return x
