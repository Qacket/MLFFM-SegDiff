import torch
torch.backends.cudnn.enabled = False
from torch import nn
from utils.tensors import default
from blocks.common_blocks import Upsample, Downsample, Residual, PreNorm
from modules.embeddings import SinusoidalPositionEmbeddings
from blocks.resnet_blocks import ResnetBlock
from blocks.attentions import Attention, LinearAttention
import torch.nn.functional as F

__all__ = ["MLFFMSegDiff", "MLFFMSeg"]

# Kaiming Initialization
class KaimingInitializedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(KaimingInitializedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # 使用 Kaiming 初始化方法初始化卷积层的权重
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


# XavierInitialization
class XavierInitializedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(XavierInitializedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # 使用 Xavier 初始化方法初始化卷积层的权重
        nn.init.xavier_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = KaimingInitializedConv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = XavierInitializedConv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = XavierInitializedConv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = KaimingInitializedConv2d(in_planes, out_planes,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# 多层次特征融合，将多个层次的特征图和本层次的特征图进行融合，最后输出固定通道数 32
class SDI(nn.Module):
    def __init__(self, index, out_channels, mid_channels=32):
        super().__init__()
        self.index = index
        self.mid_channels = mid_channels
        self.convs = nn.ModuleList(
            [nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1) for _ in range(index)])
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return self.out_conv(ans)


# 跳跃连接
class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=1, attention="both"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.ca = ChannelAttention(self.in_channels) if attention in ("both", "channel") else None
        self.sa = SpatialAttention() if attention in ("both", "spatial") else None
        self.trans_layer = BasicConv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        if self.ca is not None:
            x = self.ca(x) * x
        if self.sa is not None:
            x = self.sa(x) * x
        x = self.trans_layer(x)
        return x


class RB(nn.Module):
    """Some Information about RB"""

    def __init__(self, dim_in, dim_out, time_dim, resnet_block_groups=8):
        super().__init__()
        self.rb = ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups)

    def forward(self, x, t):
        return self.rb(x, t)


class LAtt(nn.Module):
    """Some Information about LAtt"""

    def __init__(self, dim, bn=True):
        super().__init__()
        self.att = Residual(PreNorm(dim, LinearAttention(dim)))
        self.bn = nn.BatchNorm2d(dim) if bn else nn.Identity()

    def forward(self, x):
        return self.bn(self.att(x))


class LSAtt(nn.Module):
    '''Linear-Self Attention
    '''

    def __init__(self, dim, latt_bn=True):
        super().__init__()
        self.s_attn = Residual(PreNorm(dim, Attention(dim)))
        self.l_attn = LAtt(dim, bn=latt_bn)

    def forward(self, x):
        x_s = self.s_attn(x)
        x_l = self.l_attn(x)
        return x_s + x_l


class BM(nn.Module):
    '''Bottleneck Module
    '''

    def __init__(self, dim_in, dim_out, time_dim, resnet_block_groups=4, latt_bn=True):
        super().__init__()

        self.rb_before = RB(dim_in, dim_out, time_dim, resnet_block_groups)
        self.ls_attn = LSAtt(dim_out, latt_bn=latt_bn)
        self.rb_after = RB(2 * dim_out, dim_out, time_dim, resnet_block_groups)

    def forward(self, x, t):
        x = self.rb_before(x, t)
        x_a = self.ls_attn(x)
        x = torch.cat((x, x_a), dim=1)
        x = self.rb_after(x, t)
        return x


class EM(nn.Module):
    """Some Information about EM"""

    def __init__(self, dim_x, dim_g, time_x, time_g, resnet_block_groups=8, latt_bn=True):
        super().__init__()
        self.rb_x1 = RB(dim_x, dim_x, time_x, resnet_block_groups)
        self.rb_g1 = RB(dim_g, dim_g, time_g, resnet_block_groups)
        self.rb_x2 = RB(dim_x, dim_x, time_x, resnet_block_groups)
        self.rb_g2 = RB(dim_g + dim_x, dim_g, time_g, resnet_block_groups)

        self.g_feedback = nn.Conv2d(dim_g, dim_x, 1)
        self.att_x = LAtt(dim_x, bn=latt_bn)
        self.att_g = LAtt(dim_g + dim_x, bn=latt_bn)

    def forward(self, x, g, t_x, t_g):
        x = self.rb_x1(x, t_x)

        g = self.rb_g1(g, t_g)
        g = torch.cat((x, g), dim=1)
        h = g.clone()  # skip

        g = self.rb_g2(g, t_g)
        x = self.rb_x2(self.g_feedback(g) * x, t_x)

        x = self.att_x(x)
        g = torch.cat((x, g), dim=1)
        b = g.clone()  # skip

        g = self.att_g(g)

        outputs = (x, g)
        skips = (h, b)
        return outputs, skips


class DM(nn.Module):
    """Some Information about DM"""

    def __init__(self, dim_in, dim_out, time_dim, resnet_block_groups=8, latt_bn=True):
        super().__init__()

        self.rb_d1 = RB(dim_in, dim_out, time_dim, resnet_block_groups)
        self.rb_d2 = RB(dim_in, dim_out, time_dim, resnet_block_groups)
        self.att_x = LAtt(dim_out, bn=latt_bn)

    def forward(self, x, b, h, t):
        x = torch.cat((x, b), dim=1)
        x = self.rb_d1(x, t)
        x = torch.cat((x, h), dim=1)
        x = self.rb_d2(x, t)
        x = self.att_x(x)
        return x


class MLFFMSegDiff(nn.Module):
    def __init__(
            self,
            dim_x, dim_g,
            channels_x=1, channels_g=3,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            dim_x_mults=None,
            dim_g_mults=None,
            resnet_block_groups=4,
            mlffm=True,
            skip_attention="both",
            cross_level_fusion=True,
            latt_bn=True,
            **kwargs
    ):
        '''
        `x` and `g` refer to target and guidance correspondingly.

        Ablation switches (defaults = full model):
        - mlffm: False passes encoder skips to the decoder unchanged, as DermoSegDiff does;
        - skip_attention: attention in each SkipBlock, "both", "channel", "spatial" or "none";
        - cross_level_fusion: False lets each SDI use only its own level;
        - latt_bn: BatchNorm after every linear attention (LAtt); DermoSegDiff has none.
        '''
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected model params: {sorted(kwargs)}")
        if skip_attention not in ("both", "channel", "spatial", "none"):
            raise ValueError(f"skip_attention must be both, channel, spatial or none, got {skip_attention!r}")
        if not mlffm and (skip_attention != "both" or not cross_level_fusion):
            raise ValueError("skip_attention and cross_level_fusion have no effect when mlffm is False")
        self.mlffm = mlffm
        self.cross_level_fusion = cross_level_fusion

        # determine dimensions
        init_dim_x = default(init_dim, dim_x)
        init_dim_g = default(init_dim, dim_g)

        dims_x = [init_dim_x, *map(lambda m: dim_x * m, default(dim_x_mults, dim_mults))]
        dims_g = [init_dim_g, *map(lambda m: dim_g * m, default(dim_g_mults, dim_mults))]

        in_out_x = list(zip(dims_x[:-1], dims_x[1:]))
        in_out_g = list(zip(dims_g[:-1], dims_g[1:]))

        num_resolutions = len(in_out_x)

        self.encoder_blocks = nn.ModuleList([])

        self.skip_blocks_b, self.sdi_blocks_b = nn.ModuleList([]), nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])

        # time embeddings
        time_dim_x = dim_x * 4
        self.time_dim_x = time_dim_x
        self.time_mlp_x = nn.Sequential(
            SinusoidalPositionEmbeddings(dim_x),  # (batch,dim)
            nn.Linear(dim_x, time_dim_x),
            nn.GELU(),
            nn.Linear(time_dim_x, time_dim_x),
        )
        time_dim_g = dim_g * 4
        self.time_dim_g = time_dim_g
        self.time_mlp_g = nn.Sequential(
            SinusoidalPositionEmbeddings(dim_g),  # (batch,dim)
            nn.Linear(dim_g, time_dim_g),
            nn.GELU(),
            nn.Linear(time_dim_g, time_dim_g),
        )

        # initial steps
        self.init_conv_x = nn.Conv2d(channels_x, init_dim_x, 1, padding=0)
        self.init_conv_g = nn.Conv2d(channels_g, init_dim_g, 1, padding=0)

        # building encoder
        for ind, ((dim_in_x, dim_out_x), (dim_in_g, dim_out_g)) in enumerate(zip(in_out_x, in_out_g)):
            is_last = ind >= (num_resolutions - 1)

            encoder = EM(dim_x=dim_in_x, dim_g=dim_in_g, time_x=time_dim_x, time_g=time_dim_g,
                         resnet_block_groups=resnet_block_groups, latt_bn=latt_bn)

            if mlffm:
                skip_block_b = SkipBlock(dim_in_x + dim_in_g, attention=skip_attention)
                self.skip_blocks_b.append(skip_block_b)
                # TODO mid_channel 参数可以修改，属于超参数，后期实验验证。
                sdi_b = SDI(index=6 if cross_level_fusion else 1,
                            out_channels=dim_in_x + dim_in_g, mid_channels=32)
                self.sdi_blocks_b.append(sdi_b)

            g_down = nn.Conv2d(dim_in_g + dim_in_x, dim_out_g, 3, padding=1) if is_last else Downsample(
                dim_in_g + dim_in_x, dim_out_g)

            x_down = nn.Conv2d(dim_in_x, dim_out_x, 3, padding=1) if is_last else Downsample(dim_in_x, dim_out_x)
            self.encoder_blocks.append(nn.ModuleList([encoder, g_down, x_down]))

        # building bottleneck
        self.bottleneck = BM(dim_in=dims_x[-1] + dims_g[-1], dim_out=dims_x[-1], time_dim=time_dim_x,
                             latt_bn=latt_bn)

        # building decoder
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out_x)):
            is_last = ind == (len(in_out_x) - 1)
            skip_in = list(reversed(in_out_g))[ind][0] + dim_in

            decoder = DM(dim_in=skip_in + dim_out, dim_out=dim_out, time_dim=time_dim_x,
                         resnet_block_groups=resnet_block_groups, latt_bn=latt_bn)
            d_up = nn.Conv2d(dim_out, dim_in, 3, padding=1) if is_last else Upsample(dim_out, dim_in)
            self.decoder_blocks.append(nn.ModuleList([decoder, d_up]))

        # final steps
        self.final_rb = RB(2 * dim_x + dim_g, dim_x, time_dim_x, resnet_block_groups=1)
        self.final_conv = nn.Conv2d(dim_x, default(out_dim, channels_x), 1)

    def fuse_skips(self, skips_b):
        """MLFFM on the encoder skips `b` (one per level); returns the skips given to the decoder."""
        if not self.mlffm:
            return skips_b
        new_skips_b = [skip_b(b) for b, skip_b in zip(skips_b, self.skip_blocks_b)]
        if self.cross_level_fusion:
            return [sdi_b(new_skips_b, b) for b, sdi_b in zip(new_skips_b, self.sdi_blocks_b)]
        return [sdi_b([b], b) for b, sdi_b in zip(new_skips_b, self.sdi_blocks_b)]

    def forward(self, x, g, time):
        x = self.init_conv_x(x)
        r_x = x.clone()

        g = self.init_conv_g(g)
        r_g = g.clone()

        t_x = self.time_mlp_x(time)
        t_g = self.time_mlp_g(time)

        skips_h, skips_b = [], []
        for (encoder, g_down, x_down) in self.encoder_blocks:
            (x, g), (h, b) = encoder(x, g, t_x, t_g)
            skips_h.append(h)
            skips_b.append(b)
            g = g_down(g)
            x = x_down(x)

        x = self.bottleneck(torch.cat((x, g), dim=1), t_x)
        final_skips = list(zip(skips_h, self.fuse_skips(skips_b)))

        for (decoder, d_up) in self.decoder_blocks:
            (h, b) = final_skips.pop()
            x = decoder(x, b, h, t_x)
            x = d_up(x)

        x = self.final_rb(torch.cat((x, r_x, r_g), dim=1), t_x)
        return self.final_conv(x)


class MLFFMSeg(MLFFMSegDiff):
    """Direct-mask counterpart used by the non-diffusion ablation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.time_mlp_x
        del self.time_mlp_g

    def forward(self, image):
        zero_mask = image.new_zeros(
            image.shape[0], self.init_conv_x.in_channels, image.shape[-2], image.shape[-1]
        )
        x = self.init_conv_x(zero_mask)
        r_x = x.clone()
        g = self.init_conv_g(image)
        r_g = g.clone()
        t_x = image.new_zeros((image.shape[0], self.time_dim_x))
        t_g = image.new_zeros((image.shape[0], self.time_dim_g))

        skips_h, skips_b = [], []
        for encoder, g_down, x_down in self.encoder_blocks:
            (x, g), (h, b) = encoder(x, g, t_x, t_g)
            skips_h.append(h)
            skips_b.append(b)
            g = g_down(g)
            x = x_down(x)

        x = self.bottleneck(torch.cat((x, g), dim=1), t_x)
        final_skips = list(zip(skips_h, self.fuse_skips(skips_b)))

        for decoder, d_up in self.decoder_blocks:
            h, b = final_skips.pop()
            x = decoder(x, b, h, t_x)
            x = d_up(x)

        x = self.final_rb(torch.cat((x, r_x, r_g), dim=1), t_x)
        return self.final_conv(x)
