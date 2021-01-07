import torch
import torch.nn as nn

from enum import Enum


class DownSamplingType(Enum):
    MAX_POOL = 0
    MAX_POOL_KEEP_INDICES = 1
    STRIDED_CONV = 2


class UpSamplingType(Enum):
    UPSAMPLE = 0
    MAX_UNPOOL = 1
    STRIDED_CONV_TRANSPOSE = 2


class MultiConv(nn.Module):
    def __init__(self, channel_counts, kernel_size=3, padding=1):
        super(MultiConv, self).__init__()
        self.convs = nn.Sequential(
            *sum([
                [nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding)]
                + [nn.BatchNorm2d(out_dim)] + [nn.ReLU()]
                for in_dim, out_dim in zip(channel_counts, channel_counts[1:])
            ], [])
        )

    def forward(self, x):
        return self.convs(x)


class MultiConvTranspose(nn.Module):
    def __init__(self, channel_counts, kernel_size=3, padding=1, remove_last_activation=False):
        super(MultiConvTranspose, self).__init__()
        self.deconvs = sum([
                [nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding)]
                + [nn.BatchNorm2d(out_dim)] + [nn.ReLU()]
                for in_dim, out_dim in zip(channel_counts, channel_counts[1:])
        ], [])
        if remove_last_activation:
            del self.deconvs[-1]

        self.deconvs = nn.Sequential(*self.deconvs)

    def forward(self, x):
        return self.deconvs(x)


class Encoder(nn.Module):
    def __init__(self, in_features, out_features, n_blocks,
                 kernel_size=3, padding=1, downsampling_op=DownSamplingType.MAX_POOL):
        super(Encoder, self).__init__()
        channel_counts = [in_features] + [out_features] * n_blocks
        self.convs = MultiConv(channel_counts, kernel_size, padding)
        if downsampling_op == DownSamplingType.MAX_POOL:
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsampling_op == DownSamplingType.MAX_POOL_KEEP_INDICES:
            self.down = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        elif downsampling_op == DownSamplingType.STRIDED_CONV:
            self.down = nn.Conv2d(out_features, out_features, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # returns the output and the indices after the pooling
        return self.down(self.convs(x))


class Decoder(nn.Module):
    def __init__(self, in_features, out_features, n_blocks,
                 kernel_size=3, padding=1, upsampling_op=UpSamplingType.UPSAMPLE, remove_last_activation=False):
        super(Decoder, self).__init__()
        channel_counts = [in_features] * n_blocks + [out_features]
        self.deconvs = MultiConvTranspose(channel_counts, kernel_size, padding, remove_last_activation)
        if upsampling_op == UpSamplingType.UPSAMPLE:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif upsampling_op == UpSamplingType.MAX_UNPOOL:
            self.up = nn.MaxUnpool2d(kernel_size=2, stride=2)
        elif upsampling_op == upsampling_op.STRIDED_CONV_TRANSPOSE:
            self.up = nn.ConvTranspose2d(in_features, in_features, kernel_size=2, stride=2)

    def forward(self, x, indices = None, output_size = None):
        if isinstance(self.up, nn.MaxUnpool2d):
            assert indices is not None and output_size is not None
            x = self.up(x, indices, output_size=output_size)
        else:
            x = self.up(x)
        return self.deconvs(x)

