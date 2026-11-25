import torch
import torch.nn as nn

def img_to_patches(tensor, patch_size):
    # Assume H and W are divisible by patch_size

    B, C, H, W = tensor.shape
    P = patch_size  # P = Ph = Pw
    D = C * P * P  # Embedding dimension

    # Calculate the number of patches
    Nh = H // P
    Nw = W // P
    L = Nh * Nw  # Sequence length

    # 1. and 2. Reshape and Permute to group patches and embedding components
    tensor = tensor.reshape(B, C, Nh, P, Nw, P)
    tensor = tensor.permute(0, 2, 4, 1, 3, 5) # (B, Nh, Nw, C, P, P)

    # 3. Flatten C*P*P into D
    tensor = tensor.reshape(B, Nh, Nw, D) # (B, Nh, Nw, D)

    # 4. Flatten Nh*Nw into L (Sequence Length)
    tensor = tensor.reshape(B, L, D) # (B, L, D)

    # 5. Permute to final shape (L, B, D)
    return tensor.permute(1, 0, 2) # (L, B, D)

def patches_to_img(
    patch_embeddings: torch.Tensor,
    original_channels: int,
    original_height: int,
    original_width: int,
    patch_size: int
) -> torch.Tensor:
    """
    Reverts a patch-embedded tensor (L, B, D) back to its original image shape (B, C, H, W).

    This function reverses the typical Vision Transformer patching process, assuming 
    the spatial dimensions (H, W) were perfectly divisible by the patch_size.

    Args:
        patch_embeddings (torch.Tensor): The input tensor of shape (L, B, D), 
                                         where L is sequence length, B is batch size, 
                                         and D is embedding dimension.
        original_channels (int): The original number of image channels (C).
        original_height (int): The original image height (H).
        original_width (int): The original image width (W).
        patch_size (int): The height and width of the square patches (P).

    Returns:
        torch.Tensor: The reconstructed image tensor of shape (B, C, H, W).
    """
    
    # ------------------------------------------------
    # 1. Reverse Permute: (L, B, D) -> (B, L, D)
    # The forward pass ended with a permute to (L, B, D), so we reverse it.
    # ------------------------------------------------
    tensor = patch_embeddings.permute(1, 0, 2)
    # New shape: (B, L, D) = (B, Sequence Length, Embedding Dimension)
    B, L, D = tensor.shape

    # ------------------------------------------------
    # 2. Calculate known spatial dimensions from original shape
    # ------------------------------------------------
    P = patch_size
    C = original_channels
    H = original_height
    W = original_width
    
    # Check consistency: D must equal C * P * P
    expected_D = C * P * P
    if D != expected_D:
        raise ValueError(
            f"Dimension mismatch: Embedding dimension D={D} should equal C*P*P={expected_D}. "
            f"Check your inputs for C={C}, P={P}."
        )

    # Calculate number of patches in height (Nh) and width (Nw)
    Nh = H // P
    Nw = W // P
    
    # Check consistency: L must equal Nh * Nw
    expected_L = Nh * Nw
    if L != expected_L:
        raise ValueError(
            f"Sequence length L={L} should equal Nh*Nw={expected_L}. "
            f"Check your inputs for H={H}, W={W}, P={P}."
        )

    # ------------------------------------------------
    # 3. Reverse Flatten L: (B, L, D) -> (B, Nh, Nw, D)
    # Unflatten L back into (Nh, Nw)
    # ------------------------------------------------
    tensor = tensor.reshape(B, Nh, Nw, D)
    
    # ------------------------------------------------
    # 4. Reverse Flatten D: (B, Nh, Nw, D) -> (B, Nh, Nw, C, P, P)
    # Unflatten D back into (C, P, P)
    # ------------------------------------------------
    tensor = tensor.reshape(B, Nh, Nw, C, P, P)
    
    # ------------------------------------------------
    # 5. Reverse Permute: (B, Nh, Nw, C, P, P) -> (B, C, Nh, P, Nw, P)
    # Reverse the permutation to align dimensions for final reshape
    # Original forward step was: permute(0, 2, 4, 1, 3, 5) 
    # The reverse permute is the inverse index mapping.
    # From (B, Nh, Nw, C, P, P) to (B, C, Nh, P, Nw, P)
    # The forward indices were 0, 2, 4, 1, 3, 5 -> indices 0, 1, 2, 3, 4, 5 were mapped to these locations.
    # The reverse mapping is: 0->0, 3->1, 1->2, 4->3, 2->4, 5->5
    # Let's verify: [B, C, Nh, P, Nw, P] -> [0, 3, 1, 4, 2, 5]
    # No, that's complex. Let's trace the dimensions:
    # Target: [B, C, Nh, P, Nw, P]
    # Source: [B, Nh, Nw, C, P, P]
    # Indices mapping: [0, 3, 1, 4, 2, 5]
    # B (0) -> 0
    # C (3) -> 1
    # Nh (1) -> 2
    # P (4) -> 3
    # Nw (2) -> 4
    # P (5) -> 5 
    # Yes, the indices of the source tensor that correspond to the target order are [0, 3, 1, 4, 2, 5].
    # ------------------------------------------------
    tensor = tensor.permute(0, 3, 1, 4, 2, 5) 
    # New shape: (B, C, Nh, P, Nw, P)
    
    # ------------------------------------------------
    # 6. Reverse Reshape: (B, C, Nh, P, Nw, P) -> (B, C, H, W)
    # Merge (Nh, P) to H and (Nw, P) to W
    # ------------------------------------------------
    output_tensor = tensor.reshape(B, C, H, W)
    
    return output_tensor

class TFEncoder(nn.Module):
    def __init__(self, patch_size=16, 
                 heads=32,
                 activation=nn.SiLU(),
                 channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = channels * patch_size**2
        self.mha = nn.MultiheadAttention(self.seq_len, heads)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        #self.norm0 = torch.nn.LayerNorm(patch_size**2)
        #self.norm1 = torch.nn.LayerNorm(patch_size**2)

    def forward(self, x):
        x_patched = img_to_patches(x, self.patch_size)
        self.mha(x_patched, x_patched, x_patched, need_weights=False)[0]



