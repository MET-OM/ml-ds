import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Basic ConvBlock
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization=None, activation='relu', dropout_rate=0.0, dropout_variant='spatial'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        if normalization == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization == 'layer':
            self.norm = nn.LayerNorm([out_channels, 1, 1])  # approximate
        else:
            self.norm = None
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)  # default
        
        self.dropout_rate = dropout_rate
        self.dropout_variant = dropout_variant
        if dropout_rate > 0:
            if dropout_variant == 'spatial':
                self.dropout = nn.Dropout2d(dropout_rate)
            else:
                self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        return out

# -------------------------
# Residual Block
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, channels, normalization=None, activation='relu', dropout_rate=0.0, dropout_variant='spatial'):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, normalization, activation, dropout_rate, dropout_variant)
        self.conv2 = ConvBlock(channels, channels, normalization, activation, dropout_rate, dropout_variant)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))  # simple residual connection

# -------------------------
# ResNet Backbone
# -------------------------
class ConvResNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=8, n_blocks=8, normalization=None,
                 dropout_rate=0.0, dropout_variant='spatial', attention=False,
                 activation='relu', localcon_layer=True):
        super().__init__()
        
        self.localcon_layer = None
        if localcon_layer:
            self.localcon_layer = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
            in_channels = n_filters
        
        self.initial_conv = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResBlock(n_filters, normalization, activation, dropout_rate, dropout_variant) 
            for _ in range(n_blocks)
        ])
        
        # Optional attention placeholder
        self.attention = None
        if attention:
            # Can implement e.g., channel attention here
            pass
        
        self.final_conv = nn.Conv2d(n_filters, out_channels, kernel_size=1)
    
    def forward(self, x):
        if self.localcon_layer:
            x = self.localcon_layer(x)
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.attention:
            x = self.attention(x)
        x = self.final_conv(x)
        return x
