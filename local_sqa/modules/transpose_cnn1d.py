import torch
from torch import nn

from padertorch.contrib.je.modules.conv import CNN1d


class TransposeCNN1d(nn.Module):
    """
    wraps padertorch CNN1d but accepts (N, T, C) and internally converts to (N, C, T)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_layer=False,
        output_layer=False,
        activation_fn="leaky_relu",
        norm=None,
        pre_activation=False,
        **kwargs,
    ):
        super().__init__()
        self.cnn = CNN1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            input_layer=input_layer,
            output_layer=output_layer,
            activation_fn=activation_fn,
            norm=norm,
            pre_activation=pre_activation,
            **kwargs,
        )

    def forward(self, x, sequence_lengths=None, state=None):
        # x: (N, T, C) -> (N, C, T)
        x = x.transpose(1, 2)

        out = self.cnn(x, sequence_lengths=sequence_lengths, state=state)

        # be robust if CNN1d returns (y, seq_len) or (y, seq_len, new_state)
        if isinstance(out, (tuple, list)):
            if len(out) == 2:
                y, seq_len = out
                y = y.transpose(1, 2)  # (N, C, T) -> (N, T, C)
                return y, seq_len
            if len(out) == 3:
                y, seq_len, new_state = out
                y = y.transpose(1, 2)
                return y, seq_len, new_state

        # fallback (unexpected)
        y = out.transpose(1, 2)
        return y, sequence_lengths
