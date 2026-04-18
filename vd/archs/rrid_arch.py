


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter
import numbers
import einops
from einops import rearrange
from spikingjelly.activation_based.neuron import (
    LIFNode, IFNode, ParametricLIFNode
)
from spikingjelly.activation_based import neuron, functional, layer, surrogate
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile
v_th = 0.2

alpha = 1 / (2 ** 0.5)
from basicsr.archs.arch_util import to_2tuple, trunc_normal_


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=((kernel_size - 1) * dilation // 2), bias=bias, stride=stride, dilation=dilation)


class SpikingDownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SpikingDownSample, self).__init__()
        functional.set_step_mode(self, step_mode='m')

        self.down = nn.Sequential(
            # 下采样
            layer.AvgPool2d(kernel_size=2, stride=2, step_mode='m'),
            
            # 脉冲激活
            LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False),
            
            # 升通道: C -> C + s_factor
            layer.Conv2d(in_channels, in_channels + s_factor, kernel_size=1, stride=1, padding=0, 
                         step_mode='m', bias=False),
            
            # BN
            # layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th,
            #                                     num_features=in_channels + s_factor, affine=True),
        )

    def forward(self, x):
        return self.down(x)


class SpikingSkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SpikingSkipUpSample, self).__init__()
        functional.set_step_mode(self, step_mode='m')

        self.up = nn.Sequential(
            # 上采样尺寸 ×2
            layer.Upsample(scale_factor=2, mode='bilinear', align_corners=False, step_mode='m'),

            # 激活脉冲
            LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False),

            # 通道调整: in_channels + s_factor → in_channels
            layer.Conv2d(in_channels + s_factor, in_channels, kernel_size=1, stride=1, padding=0, 
                         step_mode='m', bias=False),

            # BN
            # layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=in_channels, affine=True),
        )

    def forward(self, x, y):
        x = self.up(x)

        # 尺寸安全检查（可选）
        if x.shape != y.shape:
            x = functional.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False)

        return x + y



class UpSampling(nn.Module):
    def __init__(self, dim):
        super(UpSampling, self).__init__()
        self.scale_factor = 2
        self.up = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode='m', bias=False),
            # layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim // 2,
            #                                     affine=True),
        )

    def forward(self, input):
        temp = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor,
                            input.shape[4] * self.scale_factor)).cuda()
        # print(temp.device,'-----')
        output = []
        for i in range(input.shape[0]):
            # temp[i] = self.up(input[i])
            # print(input[i].shape)
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode='bilinear')
            # print(temp.shape)
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        return self.up(out)

class Spiking_Block(nn.Module):
    def __init__(self, dim, v_th=v_th, alpha=alpha):
        super(Spiking_Block, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.residual = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m'),
            # layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),

            LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False,
                         step_mode='m'),
            # layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th * 0.5, affine=True),
        )
        self.attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=20, kernel_size=3, C=dim)

    def forward(self, x):
        f = x.clone() # Avoid modifying x in place
        shortcut = f.clone()    # Clone f to avoid modifying it in place later
        out = self.residual(f)
        out = self.attn(out) + shortcut  # Ensure no in-place modification here
        return out



class CLIF(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, dilation=1):
        super(CLIF, self).__init__()

        self.CA = Spiking_Block(dim= n_feat)

    def forward(self, x):
        res = self.CA(x)
        res += x
        return res




class SkipUpSample(nn.Module):
    # 定义一个类SkipUpSample，继承自nn.Module
    def __init__(self, in_channels, s_factor):
        # 初始化函数，接收输入通道数和上采样因子
        super(SkipUpSample, self).__init__()
        # 调用父类的初始化函数
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

        # 定义一个上采样操作，包括Upsample和Conv2d
    def forward(self, x, y):
        # 定义前向传播函数，接收输入x和y
        x = self.up(x)
        # 对输入x进行上采样操作
        x = x + y
        # 将上采样后的x与输入y相加
        return x




class Encoder_1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Encoder_1, self).__init__()
        functional.set_backend(self, backend='cupy')
        functional.set_step_mode(self, step_mode='m')
        self.encoder_level1 = [CLIF(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CLIF(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.encoder_level3 = [CLIF(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)


        self.down12 = SpikingDownSample(n_feat, scale_unetfeats)
        self.down23 = SpikingDownSample(n_feat + scale_unetfeats, scale_unetfeats)


    def forward(self, x):
        enc1 = self.encoder_level1(x)

        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)

        return [enc1, enc2, enc3]


class Decoder_1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, pool='avg'):
        super(Decoder_1, self).__init__()

        self.decoder_level1 = [CLIF(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CLIF(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.decoder_level3 = [CLIF(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.up21 = SpikingSkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SpikingSkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, enc2)
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, enc1)
        dec1 = self.decoder_level1(x)



        return [dec1, dec2, dec3]


class Encoder_2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Encoder_2, self).__init__()

        self.encoder_level1 = [CLIF(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CLIF(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.encoder_level3 = [CLIF(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = SpikingDownSample(n_feat, scale_unetfeats)
        self.down23 = SpikingDownSample(n_feat + scale_unetfeats, scale_unetfeats)


    def forward(self, x):
        enc1 = self.encoder_level1(x)
       
        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
       
        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        

        return [enc1, enc2, enc3]


class Decoder_2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, pool='avg'):
        super(Decoder_2, self).__init__()

        self.decoder_level1 = [CLIF(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CLIF(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.decoder_level3 = [CLIF(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)


        self.up21 = SpikingSkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SpikingSkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, enc2)
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, enc1)
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


class Cross_Attention(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = 4
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))
        # self.upsample = nn.Sequential(conv(n_feat, n_feat * 4, kernel_size, bias=bias), nn.PReLU(),
        #                           nn.PixelShuffle(2),
        #                           conv(n_feat, n_feat, kernel_size, bias=bias))

        # self.upsample = nn.Sequential(conv(n_feat, n_feat * 4, 3, bias=False), nn.PReLU(),
        #                               nn.PixelShuffle(2),
        #                               conv(n_feat, n_feat, 3, bias=False))
        self.conv_raw = nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, bias=bias)
        self.conv_rgb = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias, groups=n_feat, padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, rgb, raw):
        # raw_up = self.upsample(raw)
        qk = self.conv_raw(raw)
        v = self.conv_rgb(rgb)

        _, _, h, w = qk.shape
        q, k = qk.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.feedforward(out + raw)
        return out






class dct_weight(nn.Module):
    def __init__(self, in_c=40, out_c=40):
        super(dct_weight, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class directional_dct_layer(nn.Module):
    def __init__(self, in_c=40, out_c=40):
        super(directional_dct_layer, self).__init__()
        self.ddct_conv = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.GELU()
        self.conv_dct = nn.Conv2d(in_c, out_c, kernel_size=3, stride=8, padding=2, dilation=2, groups=in_c)

    def forward(self, x):
        out = self.act(self.ddct_conv(x)) + x
        out = self.conv_dct(out)
        return out


def check_image_size(h, w, bs):
    new_h = h
    new_w = w
    if h % bs != 0:
        new_h = h + (bs - h % bs)
    if w % bs != 0:
        new_w = w + (bs - w % bs)
    return new_h, new_w


def IDDCTmode0(im):
    C, M, N = im.shape
    DD = torch.zeros((C, M, N))
    D = torch.zeros((C, M, N))

    for i in range(N):
        DD[:, :, i] = torch.fft.ifft(im[:, :, i], norm='ortho', dim=-1).real

    for j in range(M):
        D[:, j, :] = torch.fft.ifft(DD[:, j, :], norm='ortho', dim=-1).real
    return D


def directional_inverse_dct_layer(img, bs=8, mode=0):
    b, ch, h, w = img.shape
    imt = img.view(b * ch, h, w)
    c, m, n = imt.shape
    new_m, new_n = check_image_size(m, n, bs)
    new_imt = torch.zeros((c, new_m, new_n)).cuda()
    new_imt[:, :m, :n] = imt
    imf = torch.zeros((c, new_m, new_n)).cuda()
    for ii in range(0, new_m, bs):
        for jj in range(0, new_n, bs):
            cb = new_imt[:, ii:ii + bs, jj:jj + bs]
            # CB = DDCT_transform(cb, mode)
            # cbf = IDDCT(CB, mode)
            CB = cb
            cbf = IDDCTmode0(CB)
            imf[:, ii:ii + bs, jj:jj + bs] = cbf
    imf = imf[:, :m, :n]
    didct = imf.view(b, ch, h, w)
    return didct


class FSM(nn.Module):
    def __init__(self, num_feat=40, block_size=8):
        super(FSM, self).__init__()
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_dct_0 = directional_dct_layer(in_c=num_feat, out_c=num_feat)

        self.dct_weight = dct_weight(in_c=num_feat, out_c=num_feat)
        self.in_c = num_feat
        self.bs = block_size

        self.after_rdct = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        _, _, h, w = x.size()
        dct_feat = self.act(self.conv(x))
        # mode 0
        dct_feat_0 = self.conv_dct_0(dct_feat)
        out_0 = directional_inverse_dct_layer(dct_feat_0, bs=self.bs, mode=0)
        out_0 = F.interpolate(out_0, size=(h, w), mode='bilinear', align_corners=False)

        dct_weight = self.dct_weight(x)
        out = torch.mul(out_0, dct_weight)

        out = self.after_rdct(out)
        return out




class GDCA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GDCA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

    
class DMSFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DMSFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        # 第一层：5x5 深度可分离 + dilation=2
        self.project_in = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=4, dilation=2, groups=dim, bias=bias),  # depthwise
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),  # pointwise
            nn.GELU()
        )

        # 第二层：3x3 深度可分离 + dilation=2
        self.project_out = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=2, dilation=2, groups=hidden_features, bias=bias),  # depthwise
            nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)  # pointwise
        )

    def forward(self, x):
        x = self.project_in(x)
        x = self.project_out(x)
        return x

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class CRM(nn.Module):
    def __init__(self, dim, num_channel_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='BiasFree'):
        super(CRM, self).__init__()



        self.channel_attn = GDCA(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = DMSFFN(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        return x

class OverlapPatchEmbed_Raw(nn.Module):
    def __init__(self, in_c=3, embed_dim=40, spike_mode="lif", LayerNorm_type='WithBias', bias=False):
        super(OverlapPatchEmbed_Raw, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias, step_mode='m')

    def forward(self, x):
        x = self.proj(x)

        return x
    
class OverlapPatchEmbed_RGB(nn.Module):
    def __init__(self, in_c=3, embed_dim=40, spike_mode="lif", LayerNorm_type='WithBias', bias=False):
        super(OverlapPatchEmbed_RGB, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=2, stride=2, bias=bias, step_mode='m')

    def forward(self, x):
        x = self.proj(x)

        return x

# RAW-RGB-Image-Demoireing Model
@ARCH_REGISTRY.register()
class DSNet(nn.Module):

    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_edecoderfeats=20,
                 kernel_size=3, reduction=4, bias=False,
                 nf=40, depths=[3, 3, 3], num_heads=[2, 2, 2], T=4
                 ):
        super(DSNet, self).__init__()
        functional.set_backend(self, backend='cupy')
        functional.set_step_mode(self, step_mode='m')
        self.T = T

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(OverlapPatchEmbed_Raw(in_c=4, embed_dim=n_feat))
        self.shallow_feat2 = nn.Sequential(OverlapPatchEmbed_RGB(in_c=3, embed_dim=n_feat))

        self.stage1_encoder = Encoder_1(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats)
        self.stage1_decoder = Decoder_1(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, pool='avg')
        self.stage1_rconv = conv(n_feat, 4, kernel_size, bias=bias)

        self.Fusion = Cross_Attention(n_feat, kernel_size, bias)
        self.tail_r = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.stage2_encoder = Encoder_2(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats)
        self.stage2_decoder = Decoder_2(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, pool='avg')
        self.stage2_rconv = conv(n_feat, n_feat, kernel_size, bias=bias)

        self.conv_first = nn.Conv2d(n_feat, nf, 2, 2)
        self.upsample = nn.Sequential(nn.Conv2d(nf, out_c * 4 ** 2, 3, 1, 1), nn.PixelShuffle(4))

        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer =CRM(nf, num_heads[i_layer], 2.66, bias=bias)
            self.layers.append(layer)
        self.norm = LayerNorm(nf, 'WithBias')


        
        self.conv_after_body = nn.Conv2d(nf, nf, 3, 1, 1)
    

    def forward_feature(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        return x

    def forward(self, lq_rgb, lq_raw):
        raw_init = lq_raw
        rgb_init = lq_rgb
         ############ Repeat Feature  ################
        if len(lq_rgb.shape) < 5:
            lq_rgb = (lq_rgb.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        if len(lq_raw.shape) < 5:
            lq_raw = (lq_raw.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        ##-------------- Branch 1: RAW---------------------
        fea_raw = self.shallow_feat1(lq_raw)
        encoder_raw = self.stage1_encoder(fea_raw)
        decoder_raw = self.stage1_decoder(encoder_raw)
        raw = decoder_raw[0].mean(0)
        lq_raw = fea_raw.mean(0)
        out_raw = self.stage1_rconv(raw) + raw_init
        namfeats_raw = raw

        ##-------------- Branch 2: RGB---------------------
        fea_rgb = self.shallow_feat2(lq_rgb)
        encoder_rgb = self.stage2_encoder(fea_rgb)
        decoder_rgb = self.stage2_decoder(encoder_rgb)
        rgb = decoder_rgb[0].mean(0)
        lq_rgb = fea_rgb.mean(0)
        c_rgb = self.stage2_rconv(rgb) + lq_rgb
        # c_rgb = self.stage2_up(c_rgb)

        ##-------------- Fusion---------------------
        fea_isp = self.Fusion(c_rgb, namfeats_raw)
        out_rgb = self.conv_first(F.relu(self.tail_r(fea_isp)))
        out_rgb = self.conv_after_body(self.forward_feature(out_rgb)) + out_rgb
        out_rgb = self.upsample(out_rgb)
        return out_rgb, out_raw


if __name__ == '__main__':
    model = DSNet().cuda()
    x = torch.rand(1, 3, 512, 512).cuda()
    y = torch.rand(1, 4, 256, 256).cuda()
    functional.set_step_mode(model, step_mode='m')
    functional.set_backend(model, backend='cupy')
    # print(model(x).shape)
    flops, params = profile(model, inputs=(x,y))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')