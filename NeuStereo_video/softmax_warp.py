import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class GeometricWarp(nn.Module):
    """
    Implements a multi-scale geometric warping layer based on the XR-Stereo paper.
    
    This module accepts a dictionary of feature tensors at different scales,
    warps each of them along with a disparity map, and returns the warped
    results in a similar dictionary structure.
    """
    def __init__(self):
        super(GeometricWarp, self).__init__()

    def forward(self, src_features_dict, src_disp, T_geo):
        """
        Warps a dictionary of source features and a disparity map.

        Args:
            src_features_dict (Dict[str, Tensor]): A dictionary where keys are scale identifiers
                (e.g., 's16', 's8') and values are feature tensors [B, C, H, W].
                One of the feature maps must have the same resolution as src_disp.
            src_disp (Tensor): Source disparity map [B, 1, H_ref, W_ref]. This defines the
                reference resolution for the geometric transformation.
            T_geo (Tensor): The 4x4 matrix to transform stereo coordinates.

        Returns:
            Tuple[Dict[str, Tensor], Tensor]:
                - A dictionary containing the warped feature tensors.
                - The warped disparity map [B, 1, H_ref, W_ref].
        """
        device = src_disp.device
        B, _, H_ref, W_ref = src_disp.shape

        # --- Step 1: Calculate projection coordinates at the reference scale ---
        y_coords_ref, x_coords_ref = torch.meshgrid(torch.arange(H_ref, device=device),
                                                    torch.arange(W_ref, device=device), indexing='ij')

        src_d_flat = src_disp.view(B, 1, -1)
        src_uv_flat = torch.stack([x_coords_ref.float().flatten(), 
                                   y_coords_ref.float().flatten()], dim=0).unsqueeze(0).repeat(B, 1, 1)
        
        src_coords_homo = torch.cat([
            src_uv_flat, src_d_flat, torch.ones(B, 1, H_ref * W_ref, device=device)], dim=1)

        dst_coords_homo = T_geo @ src_coords_homo
        
        dst_w = dst_coords_homo[:, 3:4, :].clamp(min=1e-8)
        u_proj_ref = dst_coords_homo[:, 0:1, :] / dst_w
        v_proj_ref = dst_coords_homo[:, 1:2, :] / dst_w

        # --- Step 2: Loop through each feature map and warp it ---
        warped_features_dict = {}
        
        for scale_key, src_features in src_features_dict.items():
            _, C, H, W = src_features.shape
            
            # --- Scale projection coordinates to the current feature map's resolution ---
            scale_factor_h = H / H_ref
            scale_factor_w = W / W_ref
            
            u_proj = u_proj_ref * scale_factor_w
            v_proj = v_proj_ref * scale_factor_h
            
            # --- Perform splatting for the current scale ---
            
            # Downsample disparity and importance weights to match current scale
            # We use the original disparity for importance to preserve occlusion priority
            src_disp_scaled = F.interpolate(src_disp, size=(H, W), mode='bilinear', align_corners=False)
            importance_weights = torch.exp(src_disp_scaled.view(B, 1, H * W))
            
            # This splatting logic is now scale-generic
            warped_features_dict[scale_key] = self._splat(
                src_features.view(B, C, H * W),
                importance_weights,
                u_proj.view(B, 1, H_ref, W_ref), # Pass original projection for correct sampling
                v_proj.view(B, 1, H_ref, W_ref), #
                H, W
            )

        # --- Step 3: Warp the disparity map itself at its native resolution ---
        warped_disp = self._splat(
            src_disp.view(B, 1, H_ref * W_ref),
            torch.exp(src_disp.view(B, 1, H_ref * W_ref)),
            u_proj_ref.view(B, 1, H_ref, W_ref),
            v_proj_ref.view(B, 1, H_ref, W_ref),
            H_ref, W_ref
        )

        return warped_features_dict, warped_disp

    def _splat(self, src_values, importance_weights, u_proj, v_proj, H, W):
        """A generic, vectorized splatting function."""
        B, C, _ = src_values.shape
        device = src_values.device
        # pdb.set_trace()
        
        # Resample projected coordinates to the current grid size for splatting
        u_proj_resampled = F.interpolate(u_proj, size=(H,W), mode='bilinear', align_corners=False).view(B, 1, H*W)
        v_proj_resampled = F.interpolate(v_proj, size=(H,W), mode='bilinear', align_corners=False).view(B, 1, H*W)

        # Get integer and fractional parts
        u_floor, v_floor = torch.floor(u_proj_resampled), torch.floor(v_proj_resampled)
        u_frac, v_frac = u_proj_resampled - u_floor, v_proj_resampled - v_floor

        # Vectorize coordinates and weights for 4 neighbors
        all_v = torch.cat([v_floor, v_floor, v_floor + 1, v_floor + 1], dim=2).long()
        all_u = torch.cat([u_floor, u_floor + 1, u_floor, u_floor + 1], dim=2).long()
        
        w_tl = (1 - u_frac) * (1 - v_frac); w_tr = u_frac * (1 - v_frac)
        w_bl = (1 - u_frac) * v_frac;      w_br = u_frac * v_frac
        all_w = torch.cat([w_tl, w_tr, w_bl, w_br], dim=2)
        
        splat_weights = all_w * importance_weights.repeat(1, 1, 4)

        # Create mask and gather valid indices/data
        mask = (all_u >= 0) & (all_u < W) & (all_v >= 0) & (all_v < H)
        
        B_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, -1, H * W * 4)
        valid_mask = mask.view(-1)
        valid_b = B_indices.reshape(-1)[valid_mask]
        valid_v = all_v.view(-1)[valid_mask]
        valid_u = all_u.view(-1)[valid_mask]
        
        expanded_values = src_values.repeat(1, 1, 4)
        # expanded_values = src_values.repeat_interleave(4, dim=2)
        valid_values = expanded_values.permute(0, 2, 1).reshape(-1, C)[valid_mask]
        valid_weights = splat_weights.view(-1)[valid_mask]
        
        flat_indices = valid_b * H * W + valid_v * W + valid_u

        # Perform splatting
        warped_values = torch.zeros(B * H * W, C, device=device)
        warped_values.index_add_(0, flat_indices, valid_values * valid_weights.unsqueeze(1))
        
        normalization_map = torch.zeros(B * H * W, device=device)
        normalization_map.index_add_(0, flat_indices, valid_weights)
        
        # Normalize and reshape
        normalized_values = warped_values / (normalization_map.unsqueeze(1) + 1e-8)
        return normalized_values.reshape(B, H, W, C).permute(0, 3, 1, 2)