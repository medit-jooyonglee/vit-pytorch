from models.etc.vit_pytorch.vit_pytorch import cait_3d, levit_3d, cvt_3d, twins_svt_3d
import torch
import torchinfo
x0 = torch.randn(5, 3, 48, 48, 48) 
# y0 = cait3d(x0) 
device = 'cuda'


config = dict(
image_size = 48,
patch_size = 8,
num_classes = 9,
dim = 256,
depth = 2,
cls_depth = 2,
mlp_dim = 1024,
heads = 6,
)


cait3d = cait_3d.CaiT3D(**config)    


torchinfo.summary(cait3d, input_size=[(2, 3, 48, 48, 48)], device=device)



levit3d = levit_3d.LeViT3D(
    **{**config, 'mlp_mult': 4}
)

torchinfo.summary(levit3d, input_size=[(2, 3, 48, 48, 48)], device=device)



cvt3d  = cvt_3d.CvT3D(
    **{**config, 'mlp_mult': 4}
)
torchinfo.summary(cvt3d, input_size=[(2, 3, 48, 48, 48)], device=device)




# s1_emb_dim = 64,
# s1_patch_size = 4,
# s1_local_patch_size = 7,
# s1_global_k = 7,
# s1_depth = 1,
# TwinsSVT3D doesn't take image_size/patch_size - its downsampling is driven by the
# per-stage s{1..4}_patch_size cascade, which must evenly divide the input volume at
# every stage, and s{1..4}_local_patch_size/global_k must divide (or fit within) the
# resulting per-stage resolution. The library defaults (4,2,2,2 patch / 7 local patch)
# are tuned for 224px 2D ImageNet inputs and don't fit a 48^3 volume, so override them.
# twinsvt3d = twins_svt_3d.TwinsSVT3D(
#     num_classes = config['num_classes'],
#     s1_emb_dim = 64, s1_patch_size = 4, s1_local_patch_size = 4, s1_global_k = 4, s1_depth = 1,
#     s2_emb_dim = 128, s2_patch_size = 2, s2_local_patch_size = 3, s2_global_k = 3, s2_depth = 1,
#     s3_emb_dim = 256, s3_patch_size = 2, s3_local_patch_size = 3, s3_global_k = 3, s3_depth = 5,
#     s4_emb_dim = 512, s4_patch_size = 3, s4_global_k = 1, s4_depth = 4,
# )

# patch cascade 48 -> (/3) 16 -> (/2) 8 -> (/2) 4 -> (/2) 2, each local_patch_size/global_k
# chosen to divide (or fit within) that stage's resulting resolution.
twinsvt3d = twins_svt_3d.TwinsSVT3D(
    num_classes = config['num_classes'],
    s1_emb_dim = 64, s1_patch_size = 3, s1_local_patch_size = 4, s1_global_k = 4, s1_depth = 1,
    s2_emb_dim = 128, s2_patch_size = 2, s2_local_patch_size = 4, s2_global_k = 4, s2_depth = 1,
    s3_emb_dim = 128, s3_patch_size = 2, s3_local_patch_size = 2, s3_global_k = 2, s3_depth = 2,
    s4_emb_dim = 256, s4_patch_size = 2, s4_global_k = 2, s4_depth = 2,
)

torchinfo.summary(twinsvt3d, input_size=[(2, 3, 48, 48, 48)], device=device)

