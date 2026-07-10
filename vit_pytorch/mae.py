import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64, 
        use_sparse_indices = False,
        # ndim=2,
        **kwargs
    ):
        super().__init__()
        if isinstance(encoder, nn.Module):
            pass
        else:
            from trainer import get_model
            encoder = get_model(encoder)
        # from trainer import get_model
        self.use_sparse_indices = use_sparse_indices
        
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def _sparse_indices(self, patches, device):
        # patches: (1, N, patch_dim) — batch=1 only
        # non-zero patch = at least one non-zero value in the patch
        active = patches[0].abs().sum(dim=-1) > 0  # (N,) bool
        sparse_idx = active.nonzero(as_tuple=False).squeeze(-1)  # (K,)
        num_sparse = sparse_idx.numel()

        if num_sparse < 2:
            return None, None  # caller handles this as skip

        num_masked = max(1, int(self.masking_ratio * num_sparse))
        num_masked = min(num_masked, num_sparse - 1)  # at least 1 unmasked token for encoder

        perm = torch.randperm(num_sparse, device=device)
        masked_indices   = sparse_idx[perm[:num_masked]].unsqueeze(0)   # (1, num_masked)
        unmasked_indices = sparse_idx[perm[num_masked:]].unsqueeze(0)   # (1, K-num_masked)
        return masked_indices, unmasked_indices

    def forward(self, img):
        device = img.device

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

        use_sparse_indices = self.use_sparse_indices
        if use_sparse_indices:
            assert batch == 1, "use_sparse_indices only supports batch=1"
            masked_indices, unmasked_indices = self._sparse_indices(patches, device)
            if masked_indices is None:
                return self.mask_token.sum() * 0  # too sparse — zero loss with grad_fn
        else:
            num_masked = int(self.masking_ratio * num_patches)
            rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        num_masked = masked_indices.shape[1]

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]

        encoded_tokens = self.encoder.transformer(tokens)
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss


# def restores_unpatch_embs(patch_embs, unpatch_embs, unpatch_indices, num_patches):
#     device = patch_embs.device
#     batch_range = torch.arange(patch_embs.shape[0], device=device)[:, None]
#     restored = torch.zeros(batch_range.shape[0], num_patches, patch_embs.shape[-1], device=device)
#     restored[batch_range, unpatch_indices] = unpatch_embs
#     restored[batch_range, ~unpatch_indices] = patch_embs
#     return restored

#     full_patches = torch.zeros(batch, num_patches, patch_dim, device=device)
#     full_patches[batch_range, unmasked_indices] = unmasked_patches_encoded # 원본값
#     full_patches[batch_range, masked_indices] = pred_pixel_values          # 예측값

#     # 3. Rearrange를 이용한 복원
#     # (8, 64, 3072) -> (8, 3, 256, 256)
#     recon_images = rearrange(
#         full_patches, 
#         'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
#         h=image_height // patch_height, 
#         w=image_width // patch_width, 
#         p1=patch_height, 
#         p2=patch_width
#     )
    
    
def restore_unpatchs_embs(all_patches, image_sizes, patch_sizes):
    image_height, image_width = image_sizes
    patch_height, patch_width = patch_sizes

    h = image_height // patch_height  # 256 // 32 = 8
    w = image_width  // patch_width   # 256 // 32 = 8

    # 1) 전체 64 패치 복원 (masked + unmasked 합치기)
    # all_patches = torch.zeros(batch, num_patches, pixel_values_per_patch, device=device)
    # all_patches[batch_range, unmasked_indices] = patches[batch_range, unmasked_indices]  # 원본
    # all_patches[batch_range, masked_indices]   = pred_pixel_values                        # 예측

    # 2) (8, 64, 3072) → (8, 3, 256, 256)
    reconstructed = rearrange(all_patches,
        'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
        h=h, w=w, p1=patch_height, p2=patch_width)
    return reconstructed


def to_patch_embedding(images, patch_sizes):
    patch_height, patch_width = patch_sizes
    return Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)(images)