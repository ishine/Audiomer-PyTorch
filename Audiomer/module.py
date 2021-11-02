from performer_pytorch import Performer
from performer_pytorch.performer_pytorch import FixedPositionalEmbedding
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    from .utils import MBConv
except:
    from utils import MBConv

import math


def to_channels_last(x):
    return rearrange(x, "b channels frames -> b frames channels")


def to_frames_last(x):
    return rearrange(x, "b frames channels -> b channels frames")


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, expansion_factor, use_se, equal_strides):
        super().__init__()
        context_stride = stride // 2 if equal_strides else stride
        self.to_q = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride // 2,
            bias=False,
            expansion_factor=expansion_factor,
            use_se=use_se
        )
        self.to_context = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=context_stride,
            bias=False,
            expansion_factor=expansion_factor,
            use_se=use_se
        )

    def forward(self, x):
        q = self.to_q(x)
        q = to_channels_last(q)

        context = self.to_context(x)
        context = to_channels_last(context)
        return q, context


class AudiomerEncoderBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        dim_head=64, 
        depth=1, 
        num_heads=6, 
        expansion_factor=2,
        use_attention=True,
        use_se=True,
        equal_strides=False
        ):

        super().__init__()
        stride = kernel_size - 1
        self.use_attention = use_attention

        self.conv = ConvEmbedding(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            expansion_factor=expansion_factor,
            use_se=use_se,
            equal_strides=equal_strides
        )
        if self.use_attention:
            self.performer = Performer(
                dim=out_channels,
                depth=depth,
                heads=num_heads,
                dim_head=dim_head,
                ff_glu=True,
                attn_dropout=0.2,
                use_scalenorm=True,
                ff_mult=expansion_factor
            )

    def forward(self, x):
        # (b, in_channels, input_frames) -> (b, num_frames, out_channels)
        q, context = self.conv(x)
        # (b, num_frames, out_channels) -> (b, num_frames, out_channels)
        if self.use_attention:
            out = q + self.performer(q, context=context,
                                    context_mask=torch.ones_like(context).bool())
        else:
            out = q            
        # (b, num_frames, out_channels) -> (b, out_channels, num_frames)
        out = to_frames_last(out)
        return out


class AudiomerEncoder(nn.Module):
    def __init__(self, config, kernel_sizes, dim_head, depth, num_heads, use_residual, use_cls, expansion_factor, input_size, use_se, use_attention, equal_strides, **kwargs):
        super().__init__()
        assert(len(kernel_sizes) == len(config) - 1)
        if use_cls:
            input_size += 128
        self.layers = []
        self.identity_layers = []
        self.use_residual = use_residual
        self.use_cls = use_cls
        paddings = [3, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
        for i in range(len(config) - 1):
            in_channels, out_channels = config[i], config[i+1]
            padding = paddings[i]
            self.layers.append(
                AudiomerEncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    num_heads=num_heads,
                    depth=depth,
                    dim_head=dim_head if isinstance(
                        dim_head, int) else dim_head[i],
                    expansion_factor=expansion_factor,
                    use_se=use_se, 
                    use_attention=use_attention, 
                    equal_strides=equal_strides
                )
            )
            if use_residual:
                self.identity_layers.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_sizes[i] // 2,
                            stride=kernel_sizes[i] // 2,
                            bias=False,
                            padding=padding
                        ),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
            input_size = math.floor(
                (input_size + kernel_sizes[i] // 2 + 1) / (kernel_sizes[i] // 2)) + 1

        self.layers = nn.ModuleList(self.layers)
        if use_residual:
            self.identity_layers = nn.ModuleList(self.identity_layers)
        else:
            self.identity_layers = [None] * len(self.layers)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(
                1, 1, 128))

    def forward(self, x):
        if self.use_cls:
            cls_tokens = repeat(
                self.cls_token, '() n d -> b n d', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=2)

        for (layer, id_layer) in zip(self.layers, self.identity_layers):
            x_copy = x
            x = layer(x)
            if self.use_residual:
                x_copy = id_layer(x_copy)
                x = x + x_copy
        return x


class AudiomerClassification(nn.Module):
    def __init__(self, *, config, kernel_sizes, num_classes, mlp_dim, num_heads, dim_head, depth, pool, mlp_dropout, use_residual, expansion_factor, input_size, use_attention, use_se, equal_strides):
        '''config[-1] will be the output sequence length'''
        assert(pool in ['none', "mean", "cls"])
        super().__init__()

        self.pool = pool
        self.use_cls = True if self.pool == "cls" else False

        self.encoder = AudiomerEncoder(
            config=config, kernel_sizes=kernel_sizes, num_heads=num_heads, depth=depth, use_residual=use_residual, use_cls=self.use_cls, dim_head=dim_head, expansion_factor=expansion_factor, input_size=input_size, use_attention=use_attention, use_se=use_se, equal_strides=equal_strides)

        self.classifier = nn.Sequential(
            nn.Linear(config[-1], mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.pool == "mean":
            x = x.mean(dim=2)
        else:
            x = x[:, :, 0]
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # embed = ConvEmbedding(
    #     in_channels=1,
    #     out_channels=96,
    #     stride=2,
    #     kernel_size=4,
    #     input_size=8192
    # )

    # inp = torch.randn(2, 1, 8192)
    # out = embed(inp)
    # print(out.shape)

    # inp = torch.randn(2, 1, 8191)
    # block = AudiomerEncoderBlock(
    #     in_channels=1,
    #     out_channels=96,
    #     input_size=8191
    # )
    # out = block(inp)

    # print(out.shape)

    # inp = torch.randn(2, 1, 8192)
    # block = AudiomerEncoder(
    #     input_size=8192,
    #     config=[1, 96, 96, 96, 384, 384, 384, 384],
    # )
    # out = block(inp)
    from torchsummary import summary
    # summary(block, (1, 8192), device='cpu')
    # print(out.shape)
    # config = [1, 4, 8, 16, 16, 32, 32, 64, 64, 96, 96, 192] # Audiomer L - 800K
    # config = [1, 4, 8, 16, 16, 32, 64, 128] # Audiomer M - 290K
    config = [1, 4, 8, 8, 16, 16, 32, 32, 64, 64] # Audiomer S - 180K

    input_size = 8192*1
    block = AudiomerClassification(
        input_size=input_size,
        config=config,
        kernel_sizes=[5] * (len(config) - 1),
        num_classes=12,
        depth=1,
        num_heads=2,
        pool="cls",
        mlp_dim=config[-1],
        mlp_dropout=0.2,
        use_residual=True,
        dim_head=32,
        expansion_factor=2
    ).cuda()

    inp = torch.randn(2, 1,  input_size).cuda()
    block(inp)
    summary(block, (1, input_size), device='cuda')
    count = 0
    for p in block.parameters():
        count += int(p.numel())
    print("# params: ", count)
    """
    optim = torch.optim.SGD(block.parameters(), lr=0.001)
    while True:
        inp = torch.randn(2, 1, 8192)
        target = torch.randint(0, 128, (2, 64))

        optim.zero_grad()
        out = F.log_softmax(block(inp), dim=-1)
        # print(out.shape)

        loss = 0.0
        for i in range(out.shape[1]):
            print("out",out.shape)
            print("target",target.shape)
            #loss += F.nll_loss(torch.max(out[:,i],0), target[:, i])
            loss += F.nll_loss(out[:, i, :], target[:, i])
        loss.backward()
        optim.step()
    """
    # from torchsummary import summary

    # summary(block, (1, 8192), device='cpu')
