import torch.nn as nn

from src import modules

class SegNet(nn.Module):
    def __init__(self,
                 channel_counts,
                 multiconv_lengths,
                 n_classes,
                 kernel_size = 3,
                 padding = 1,
                 downsampling_op=modules.DownSamplingType.MAX_POOL,
                 upsampling_op=modules.UpSamplingType.UPSAMPLE):
        super().__init__()

        self.downsampling_op = downsampling_op
        self.upsampling_op = upsampling_op

        assert len(channel_counts) == len(multiconv_lengths) + 1

        # The downsampling part
        self.encoders = nn.ModuleList([
            modules.Encoder(in_dim, out_dim, multiconv_length, kernel_size, padding, downsampling_op)
            for in_dim, out_dim, multiconv_length
            in zip(channel_counts, channel_counts[1:], multiconv_lengths)
        ])

        # The upsampling part
        self.decoders = nn.ModuleList([
            modules.Decoder(in_dim, out_dim, multiconv_length, kernel_size, padding, upsampling_op)
            for in_dim, out_dim, multiconv_length
            in zip(channel_counts[:1:-1], channel_counts[-2:0:-1], multiconv_lengths)
        ] + [modules.Decoder(channel_counts[1], n_classes, 2, kernel_size, padding, upsampling_op, remove_last_activation=True)]
        )

    def forward(self, x):
        # needed if upsampling_op == nn.MaxUnpool2d
        output_dims = []
        encoder_outputs = []

        # pass through encoders
        if self.upsampling_op == modules.DownSamplingType.MAX_POOL_KEEP_INDICES:
            for encoder in self.encoders:
                output_dims.append(x.size)
                x, encoder_output = encoder(x)
                encoder_outputs.append(encoder_output)
        else:
            for encoder in self.encoders:
                x = encoder(x)

        # pass through decoders
        if self.upsampling_op == modules.UpSamplingType.MAX_UNPOOL:
            for decoder, encoder_output, output_dim in zip(self.decoders, encoder_outputs[::-1], output_dims[::-1]):
                x = self.decoders(x, encoder_output, output_dim)
        else:
            for decoder in self.decoders:
                x = decoder(x)

        return x
