import torch
import torch.nn.functional as F
import numpy as np
import torchvision.ops
from torch import nn
from .ifrunet_scconv import ScConv

class SFBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            ScConv(dim),       
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, channels,out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels,out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
              )

    def forward(self, x):
        x1 = self.conv1(x)
        return x1
    
class DepthWiseConv2d(nn.Module):
    def __init__(self, channels,out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels,out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)  )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel,out_channel, kernel_size=5, padding=2, stride=1, dilation=1, groups=out_channel),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x4
    
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            DepthWiseConv2d(in_ch, out_ch),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x) 
        return x 

class UpConv2d(nn.Module):
    def __init__(self,channels,out_channel):
        super().__init__()
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels,out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel))
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels,out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
              )
 
    def forward(self,x):
        x1 = self.conv4(x)  
        x2 = self.conv5(x) + x1
        return x2

    

#encoder PatchEmbeddings
class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W

#multi-head attention
class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:#检查是否需要进行下采样（reduction）
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)#dim 是输入特征的通道数，reduction_ratio 是下采样的步幅
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        #通过查询（query）的线性变换，将输入特征 x 映射到查询矩阵 q。然后对 q 进行形状变换和维度置换，以适应后续计算
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)
        #通过键（key）和值（value）的线性变换，将下采样后的特征 x 映射到键值对 kv。对 kv 进行形状变换和维度置换，以适应后续计算    
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        #将键值对 kv 拆分为键 k 和值 v
        k, v = kv[0], kv[1]
        #计算注意力分数，通过查询 q 和键 k 的点积，然后乘以缩放因子 self.scale
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        #通过注意力权重对值 v 进行加权求和，然后对结果进行维度置换和形状变换，以得到最终的注意力输出 x_atten
        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        #通过线性变换 self.proj，将注意力输出 x_atten 映射到最终的输出特征 out
        out = self.proj(x_atten)

        return out
    
# class DeformConv2d(nn.Module):
#     def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
#         """
#         Args:
#             modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
#         """
#         super(DeformConv2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.zero_padding = nn.ZeroPad2d(padding)
#         # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
#         self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
#         # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
#         self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#         nn.init.constant_(self.p_conv.weight, 0)
#         self.p_conv.register_backward_hook(self._set_lr)
 
#         self.modulation = modulation # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
#         if modulation:
#             self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#             nn.init.constant_(self.m_conv.weight, 0)
#             self.m_conv.register_backward_hook(self._set_lr)
 
#     @staticmethod
#     def _set_lr(module, grad_input, grad_output):
#         grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
#         grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
 
#     def forward(self, x):
#         offset = self.p_conv(x)
#         if self.modulation:
#             m = torch.sigmoid(self.m_conv(x))
 
#         dtype = offset.data.type()
#         ks = self.kernel_size
#         N = offset.size(1) // 2
 
#         if self.padding:
#             x = self.zero_padding(x)
 
#         # (b, 2N, h, w)
#         p = self._get_p(offset, dtype)
 
#         # (b, h, w, 2N)
#         p = p.contiguous().permute(0, 2, 3, 1)
#         q_lt = p.detach().floor()
#         q_rb = q_lt + 1
 
#         q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
#         q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
 
#         # clip p
#         p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
 
#         # bilinear kernel (b, h, w, N)
#         g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
#         g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
#         g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
#         g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
 
#         # (b, c, h, w, N)
#         x_q_lt = self._get_x_q(x, q_lt, N)
#         x_q_rb = self._get_x_q(x, q_rb, N)
#         x_q_lb = self._get_x_q(x, q_lb, N)
#         x_q_rt = self._get_x_q(x, q_rt, N)
 
#         # (b, c, h, w, N)
#         x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
#                    g_rb.unsqueeze(dim=1) * x_q_rb + \
#                    g_lb.unsqueeze(dim=1) * x_q_lb + \
#                    g_rt.unsqueeze(dim=1) * x_q_rt
 
#         # modulation
#         if self.modulation:
#             m = m.contiguous().permute(0, 2, 3, 1)
#             m = m.unsqueeze(dim=1)
#             m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
#             x_offset *= m
 
#         x_offset = self._reshape_x_offset(x_offset, ks)
#         out = self.conv(x_offset)
 
#         return out
 
#     def _get_p_n(self, N, dtype):
#         # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
#         p_n_x, p_n_y = torch.meshgrid(
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
#         # (2N, 1)
#         p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
#         p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
 
#         return p_n
 
#     def _get_p_0(self, h, w, N, dtype):
#         # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
#         p_0_x, p_0_y = torch.meshgrid(
#             torch.arange(1, h*self.stride+1, self.stride),
#             torch.arange(1, w*self.stride+1, self.stride))
        
#         p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
 
#         return p_0
    
#     # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
#     # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
#     # pn则是p0对应卷积核每个位置的相对坐标；
#     def _get_p(self, offset, dtype):
#         N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
 
#         # (1, 2N, 1, 1)
#         p_n = self._get_p_n(N, dtype)
#         # (1, 2N, h, w)
#         p_0 = self._get_p_0(h, w, N, dtype)
#         p = p_0 + p_n + offset
#         return p
 
#     def _get_x_q(self, x, q, N):
#         # 计算双线性插值点的4邻域点对应的权重
#         b, h, w, _ = q.size()
#         padded_w = x.size(3)
#         c = x.size(1)
#         # (b, c, h*w)
#         x = x.contiguous().view(b, c, -1)
 
#         # (b, h, w, N)
#         index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
#         # (b, c, h*w*N)
#         index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
 
#         x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
 
#         return x_offset
 
#     @staticmethod
#     def _reshape_x_offset(x_offset, ks):
#         b, c, h, w, N = x_offset.size()
#         x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
#         x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
 
#         return x_offset    
    
# class DWConv(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

#     def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
#         B, N, C = x.shape
#         tx = x.transpose(1, 2).view(B, C, H, W)
#         conv_x = self.dwconv(tx)
#         return conv_x.flatten(2).transpose(1, 2) 
       
# class MixFFN(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         self.fc1 = nn.Linear(c1, c2)
#         self.dwconv = DWConv(c2)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(c2, c1)
        
#     def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
#         ax = self.act(self.dwconv(self.fc1(x), H, W))
#         out = self.fc2(ax)
#         return out
    
#TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        # self.mlp = MixFFN(dim, int(dim))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = self.norm2(tx)

        # mx = tx + self.mlp(self.norm2(tx), H, W)
        mx = mx + tx
        return mx


class Ablation(nn.Module):
    def __init__(self, channels,out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels,out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
              )

    def forward(self, x):
        x1 = self.conv1(x)
        return x1

class VicFormer(nn.Module):
    def __init__(self,in_c=3, out_c=1, dims = [64,128, 256,512], layers = [2, 2, 2, 2]):#encoder_pretrained使用预训练的参数
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [4, 4, 1, 1]
        heads = [1, 2, 4, 8]
# reduction_ratios 和 heads 参数通常不直接影响特征图的大小，
# 它们分别用于定义注意力机制中的通道数压缩比率和多头注意力的头数。
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(224, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(224//4, patch_sizes[1], strides[1],  padding_sizes[1],dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(224//8, patch_sizes[2], strides[2],  padding_sizes[2],dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(224//16, patch_sizes[3], strides[3],  padding_sizes[3],dims[2], dims[3])
        # self.deform = DeformConv2d(in_c,out_c)
        # transformer encoder
        self.block1 = nn.ModuleList([
            TransformerBlock(dims[0], heads[0], reduction_ratios[0])
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            TransformerBlock(dims[1], heads[1], reduction_ratios[1])
        for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            TransformerBlock(dims[2], heads[2], reduction_ratios[2])
        for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            TransformerBlock(dims[3], heads[3], reduction_ratios[3])
        for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

        self.f1 = nn.Sequential(
            Down(in_c, dims[0]),
        )
        self.f2 = nn.Sequential(
            Down(dims[0], dims[1]),
        )
        self.f3 = nn.Sequential(
            Down(dims[1], dims[2]),
        )
        self.f4 = nn.Sequential(
            Down(dims[2], dims[3]),
        )
        self.down = nn.MaxPool2d(2)
        self.s1 = nn.Sequential(
            DepthWiseConv2d(in_c, dims[0]),
            # Ablation(in_c, dims[0]),
            SFBlock(dims[0])
        )
        self.s2 = nn.Sequential(
            DepthWiseConv2d(dims[0],dims[1]),
            # Ablation(dims[0],dims[1]),
            SFBlock(dims[1])
        )
        self.s3 = nn.Sequential(
            DepthWiseConv2d(dims[1],dims[2]),
            # Ablation(dims[1],dims[2]),
            SFBlock(dims[2])
        )
        self.s4 = nn.Sequential(
            DepthWiseConv2d(dims[2],dims[3]),
            # Ablation(dims[2],dims[3]),
            SFBlock(dims[3])   
        )

        self.d4 = nn.Sequential(
            UpConv2d(dims[3], dims[2])
            # Ablation(dims[3], dims[2])
        )
        self.d3 = nn.Sequential(
            UpConv2d(dims[2], dims[1])
            # Ablation(dims[2], dims[1])
        )
        self.d2 = nn.Sequential(
            UpConv2d(dims[1], dims[0])
            # Ablation(dims[1], dims[0])
        )
        self.d1 = nn.Sequential(
            UpConv2d(dims[0], out_c)
            # Ablation(dims[0], out_c)
        )


    def forward(self, x):
        if x.size()[1] == 1:#？
            x = x.repeat(1, 3, 1, 1)

        #---------------Encoder-------------------------


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # outs = []
#     前向传播函数：对输入的图像进行特征提取和编码，然后通过解码器产生输出
        # stage 1
        x1, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1, H, W)
        x1 = x1 + self.norm1(x1)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 首先使用reshape方法将特征图x1的形状从(B, C, H, W)变为(B, H, W, C)，其中B表示批次大小，C表示通道数，H和W分别表示特征图的高度和宽度。
        # 然后，使用permute方法将特征图x1的第2和第3个维度进行交换，即将其形状变为(B, C, H, W)。
        # 最后，使用contiguous方法将特征图x1转换为一个连续的张量，以便于后续计算。
        # outs.append(x)
        o1 = torch.mean(x1, dim=1, keepdim=True)
        t1 = torch.mean(o1, dim=3, keepdim=True)
        # 上采样
        o1 = F.interpolate(o1, size=(224, 224), mode='bilinear', align_corners=False) 


        # stage 2
        x2, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2, H, W)
        x2 = x2 + self.norm2(x2) 
        # 128
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)
        o2 = torch.mean(x2, dim=1, keepdim=True)
        t2 = torch.mean(o2, dim=3, keepdim=True)
        # 上采样
        o2 = F.interpolate(o2, size=(224, 224), mode='bilinear', align_corners=False) 

        # stage 3
        x3, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x3 = blk(x3, H, W)
        x3 = x3 + self.norm3(x3) 
        # 256
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)
        o3 = torch.mean(x3, dim=1, keepdim=True)
        t3 = torch.mean(o3, dim=3, keepdim=True)
        # 上采样
        o3 = F.interpolate(o3, size=(224, 224), mode='bilinear', align_corners=False) 

        # stage 4
        x4, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x4 = blk(x4, H, W)
        x4 = x4 + self.norm4(x4)
        # 512
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)
        o4 = torch.mean(x4, dim=1, keepdim=True)
        # 平均池化,输出张量的维度数与输入张量相同，但被求均值的那个维度大小为1
        t4 = torch.mean(o4, dim=3, keepdim=True)
        # 上采样
        o4 = F.interpolate(o4, size=(224, 224), mode='bilinear', align_corners=False) 
        
        t = o1 + o2 + o3 + o4
        


        #---------------Decoder-------------------------     
#after trans
        out = t + x
        # transforer branch downsamping
        c0 = self.f1(out) 
        c1 = self.f2(c0) 
        c2 = self.f3(c1)
        c3 = self.f4(c2)
        # cnn branch
        y0 = self.s1(out)
        y0 = self.down(y0)
        y1 = self.s2(y0)
        y1 = self.down(y1)
        y2 = self.s3(y1)
        y2 = self.down(y2)
        y3 = self.s4(y2)
        y3 = self.down(y3)
        # fusion branch
        c0 = c0 + y0 
        c1 = c1 + y1 + t1
        c2 = c2 + y2 + t2
        c3 = c3 + y3 + t3

        
        # out4 = F.interpolate(self.d4(x3),scale_factor=(2,2),mode ='bilinear',align_corners=True) 
        out4 = F.interpolate(self.d4(c3),scale_factor=(2,2),mode ='bilinear',align_corners=True)
        o3 = out4 + c2 
        out3 = F.interpolate(self.d3(o3),scale_factor=(2,2),mode ='bilinear',align_corners=True) 
        o2 = out3 + c1
        out2 = F.interpolate(self.d2(o2),scale_factor=(2,2),mode ='bilinear',align_corners=True) 
        o1 = out2 + c0
        out1 = F.interpolate(self.d1(o1),scale_factor=(2,2),mode ='bilinear',align_corners=True) 

        return out1

