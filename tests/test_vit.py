import torch
from vit_pytorch import ViT, MAE
# from 

def test_vit():
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)
    assert preds.shape == (1, 1000), 'correct logits outputted'



def test_mae():

    ndim = 3
    patch_size = 6 if ndim == 3 else 32
    image_size = 48 if ndim == 3 else 256
    v = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        ndim=3,
        # ndi
    )

    mae = MAE(
        encoder = v,
        masking_ratio = 0.75,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        decoder_depth = 6       # anywhere from 1 to 8
    )

    if ndim == 3:
        images = torch.randn(8, 3, 48, 48, 48)
    elif ndim == 2:
        images = torch.randn(8, 3, 256, 256)

    loss = mae(images)
    loss.backward()


# test_mae()

def test_patch_embs_and_unpatch_embs():
    from vit_pytorch import mae
    
    x0 = torch.randn(8, 3, 256, 256)
    embs = mae.to_patch_embedding(x0, patch_sizes=(32, 32))
    
    assert embs.shape == (8, 64, 3072), f"Expected shape (8, 64, 3072), but got {embs.shape}"
    restored = mae.restore_unpatchs_embs(embs, image_sizes=(256, 256), patch_sizes=(32, 32))
    assert torch.allclose(x0, restored, atol=1e-6), "Restored images do not match the original images"
    
    
    
    
# test_patch_embs_and_unpatch_embs()
    
test_mae()