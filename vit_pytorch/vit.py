import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import List

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

from experiments.relative_position_embedding import Window3DRelativePositionBias
class Attention(Module):
    def __init__(self, *, dim, heads = 8, dim_head = 64, dropout = 0., 
                 relative_pos=False,
                 window_size:List[int] = None, **kwargs):
        super().__init__()
        if dim_head < 0:
            assert dim % heads == 0, 'dimension must be divisible by number of heads'
            dim_head = dim // heads
            
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.relative_pose_embeding = Window3DRelativePositionBias(window_size) if relative_pos else None
        

    def forward(self, x, x_scale=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # qk_dot = 
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.relative_pose_embeding is not None:
            dots = dots + self.relative_pose_embeding(x_scale)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head=-1, dropout = 0., 
                 relative_pos=False,
                 window_size=None, 
                 **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim=dim, heads = heads, dim_head = dim_head, dropout = dropout, 
                          relative_pos=relative_pos,  window_size=window_size, **kwargs),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, scale=None):
        for attn, ff in self.layers:
            x = attn(x, scale) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., ndim = 2, **kwargs):
        super().__init__()
        assert ndim in (2, 3), 'ndim must be 2 or 3'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if ndim == 2:
            image_height, image_width = pair(image_size)
            self.patch_size = patch_height, patch_width = pair(patch_size)

            assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

            num_patches = (image_height // patch_height) * (image_width // patch_width)
            patch_dim = channels * patch_height * patch_width

            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        else:  # ndim == 3
            image_d, image_h, image_w = (image_size, image_size, image_size) if isinstance(image_size, int) else image_size
            patch_d, patch_h, patch_w = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
            self.patch_size = (patch_d, patch_h, patch_w)

            assert image_d % patch_d == 0 and image_h % patch_h == 0 and image_w % patch_w == 0, 'Image dimensions must be divisible by the patch size.'

            num_patches = (image_d // patch_d) * (image_h // patch_h) * (image_w // patch_w)
            patch_dim = channels * patch_d * patch_h * patch_w

            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (d p0) (h p1) (w p2) -> b (d h w) (p0 p1 p2 c)', p0 = patch_d, p1 = patch_h, p2 = patch_w),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )

        num_cls_tokens = 1 if pool == 'cls' else 0

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x) + x # Residual Connection

class SimpleStem(nn.Module):
    def __init__(self, *, in_ch=1, out_ch=64, kernel_size=3, stride=2, padding=1, **kwargs):
        base_ch = out_ch // 2
        super().__init__()
        # 48 -> 24 (Stride 2) -> 12 (Stride 2)
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(inplace=True)
        )
        self.residual_block = ResidualBlock(base_ch * 2, base_ch * 2)
        
        # self.pool = lambda x: rearrange(x, 'b c d h w -> b (d h w) c')
    
    def forward(self, x):
        x = self.stem(x)
        x = self.residual_block(x)
        # x = self.pool(x)
        return x
    

def main():
    
    config = dict(
        dim=256,
        depth=2,
        heads=4,
        mlp_dim=1024, dropout = 0., 
                #  relative_pos=False,
                #  window_size
    )
    tans0 = Transformer(**config)
    x0 = torch.randn(2, 64, 256)
    y0 = tans0(x0)
    print(y0.shape)
    
    trans1 = Transformer(**{**config, 'relative_pos':True, 'window_size':[4, 4, 4]})
    # x1 = torch.randn(2, 64, 256)
    y1 = trans1(x0)
    print(y1.shape)
    scale = torch.randn(2)
    y2 = trans1(x0, scale)
    
    print(y1.shape)
    assert y0.shape == y1.shape == y2.shape 
    y3 = tans0(x0, scale)
    assert y0.shape == y3.shape
    
    
if __name__ == '__main__':
    
    stem = SimpleStem(in_ch=3, out_ch=32, kernel_size=9, stride=2)
    xs = torch.randn(1, 3, 42, 42, 42)
    # for 
    ys = stem(xs)
    print(ys.shape)
    

