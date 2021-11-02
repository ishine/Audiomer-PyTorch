import torch
from torch import nn
import torch.nn.functional as F


def make_divisible(v, divisor=8, min_value=None):
    """
    The channel number of each layer should be divisable by 8.
    The function is taken from
    github.com/rwightman/pytorch-image-models/master/timm/models/layers/helpers.py
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        out_channels: int = -1,
        **kwargs: dict,
    ):
        super(SqueezeExcitation, self).__init__()
        assert in_channels > 0

        num_reduced_channels = make_divisible(
            max(out_channels, 8) // reduction, 8
        )

        self.fc1 = nn.Conv1d(in_channels, num_reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(num_reduced_channels, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = F.adaptive_avg_pool1d(inp, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x).sigmoid()
        return x


class SepConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False
    ):
        assert(stride < kernel_size)
        super(SepConv1d, self).__init__()
        padding = kernel_size - stride#+ 1
        self.depthwise = torch.nn.Conv1d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=in_channels,
                                         bias=bias)
        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.pointwise = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=bias, padding=1)

    def forward(self, x):
        # x.shape -> (b, channels, frames)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=False,
        expansion_factor=2,
        use_se=True
    ):
        super().__init__()
        assert(stride > 1)
        self.use_se = use_se
        self.sep_conv = SepConv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, bias=bias)
        if self.use_se:
            self.se = SqueezeExcitation(
                in_channels=out_channels)

    def forward(self, inp):
        x = self.sep_conv(inp)
        if self.use_se:
            x = x * self.se(x)
        return x


if __name__ == "__main__":
    m = MBConv(1, 4, 4, stride=4, expansion_factor=8)

    inp = torch.randn(1, 1, 22050)
    out = m(inp)
    print(out.shape)
