import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricWarp(nn.Module):
    def __init__(self, alpha=-0.01):  # MUCH smaller alpha for float16 stability
        super(GeometricWarp, self).__init__()
        # Use negative alpha: exp(-0.01 * large_disp) stays in valid range
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, src_features_dict, src_disp, T_geo):
        device = src_disp.device
        B, _, H_ref, W_ref = src_disp.shape

        if T_geo.dim() == 2:
            T_geo = T_geo.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Clamp to positive values
        src_disp = src_disp.clamp(min=0.1)

        warped_features_dict = {}

        for scale_key, src_features in src_features_dict.items():
            _, C, H, W = src_features.shape
            
            u_proj, v_proj, d_proj = self._compute_projection_vectorized(
                src_disp, T_geo, H, W
            )
            
            src_disp_scaled = F.interpolate(
                src_disp, size=(H, W), mode='bilinear', align_corners=False
            )
            
            warped_features_dict[scale_key] = self._softmax_splat_vectorized(
                src_features, u_proj, v_proj, src_disp_scaled, H, W
            )

        # Warp disparity
        u_proj_ref, v_proj_ref, d_proj_ref = self._compute_projection_vectorized(
            src_disp, T_geo, H_ref, W_ref
        )
        
        d_proj_ref = d_proj_ref.clamp(min=0.1)
        
        warped_disp = self._softmax_splat_vectorized(
            d_proj_ref, u_proj_ref, v_proj_ref, src_disp, H_ref, W_ref
        )

        return warped_features_dict, warped_disp

    def _compute_projection_vectorized(self, src_disp, T_geo, H_out, W_out):
        B = src_disp.shape[0]
        device = src_disp.device
        
        d_resized = F.interpolate(
            src_disp, size=(H_out, W_out), 
            mode='bilinear', align_corners=False
        )
        
        y, x = torch.meshgrid(
            torch.arange(H_out, device=device, dtype=torch.float32),
            torch.arange(W_out, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        x = x.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        y = y.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        ones = torch.ones_like(x)
        
        coords_homo = torch.cat([x, y, d_resized, ones], dim=1)
        coords_flat = coords_homo.view(B, 4, -1)
        
        # Cast to float32 for matrix multiply if using mixed precision
        if T_geo.dtype == torch.float16:
            coords_flat = coords_flat.float()
            T_geo_compute = T_geo.float()
            transformed = torch.bmm(T_geo_compute, coords_flat)
        else:
            transformed = torch.bmm(T_geo, coords_flat)
        
        w = transformed[:, 3:4, :].clamp(min=1e-8)
        u_proj = (transformed[:, 0:1, :] / w).view(B, 1, H_out, W_out)
        v_proj = (transformed[:, 1:2, :] / w).view(B, 1, H_out, W_out)
        d_proj = (transformed[:, 2:3, :] / w).view(B, 1, H_out, W_out)
        
        # Cast back to original dtype
        if T_geo.dtype == torch.float16:
            u_proj = u_proj.half()
            v_proj = v_proj.half()
            d_proj = d_proj.half()
        
        return u_proj, v_proj, d_proj

    def _softmax_splat_vectorized(self, src_values, u_proj, v_proj, importance_map, H, W):
        B, C, _, _ = src_values.shape
        device = src_values.device
        original_dtype = src_values.dtype
        N = H * W
        
        # CRITICAL: Cast to float32 for numerical stability
        src_values = src_values.float()
        u_proj = u_proj.float()
        v_proj = v_proj.float()
        importance_map = importance_map.float()
        
        # Flatten
        src_flat = src_values.view(B, C, N)
        u_flat = u_proj.view(B, N)
        v_flat = v_proj.view(B, N)
        importance_flat = importance_map.view(B, N)
        
        # Numerically stable importance weighting
        # Normalize importance to reasonable range before exp
        importance_normalized = importance_flat / (importance_flat.max() + 1e-8)
        
        # Now exp is safe: exp(alpha * normalized) where normalized is in [0, 1]
        # With alpha=-0.01, this gives exp([-0.01, 0]) = [0.99, 1.0] - very stable!
        weights = torch.exp(self.alpha * importance_normalized)
        
        # Alternative: use softmax-style with log-sum-exp trick
        # This is even more stable for very large disparities
        # weights = torch.softmax(self.alpha * importance_flat, dim=1)
        
        # Bilinear coordinates
        u0 = torch.floor(u_flat).long()
        v0 = torch.floor(v_flat).long()
        u1 = u0 + 1
        v1 = v0 + 1
        
        wu = u_flat - u0.float()
        wv = v_flat - v0.float()
        
        # Concatenate 4 corners
        all_u = torch.cat([u0, u1, u0, u1], dim=1)
        all_v = torch.cat([v0, v0, v1, v1], dim=1)
        all_w = torch.cat([
            (1 - wu) * (1 - wv) * weights,
            wu * (1 - wv) * weights,
            (1 - wu) * wv * weights,
            wu * wv * weights
        ], dim=1)
        
        # Replicate values
        all_values = src_flat.repeat(1, 1, 4)
        
        # Valid mask
        valid = (all_u >= 0) & (all_u < W) & (all_v >= 0) & (all_v < H)
        valid_flat = valid.view(-1)
        
        # Compute global indices
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, N * 4)
        flat_indices = all_v * W + all_u
        global_indices = batch_indices * (H * W) + flat_indices
        
        # Flatten and filter
        global_indices_flat = global_indices.view(-1)[valid_flat]
        all_w_flat = all_w.view(-1)[valid_flat]
        
        # Reshape values and filter
        all_values_reshaped = all_values.permute(1, 0, 2).reshape(C, -1)
        valid_values = all_values_reshaped[:, valid_flat]
        
        # Weight the values
        weighted_values = valid_values * all_w_flat.unsqueeze(0)
        
        # Check for NaN/Inf before scatter
        if torch.isnan(weighted_values).any() or torch.isinf(weighted_values).any():
            print("WARNING: NaN/Inf in weighted_values!")
            print(f"  weights range: [{all_w_flat.min():.3f}, {all_w_flat.max():.3f}]")
            print(f"  values range: [{valid_values.min():.3f}, {valid_values.max():.3f}]")
            weighted_values = torch.nan_to_num(weighted_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scatter add
        output_flat = torch.zeros(C, B * H * W, device=device, dtype=torch.float32)
        weight_sum = torch.zeros(B * H * W, device=device, dtype=torch.float32)
        
        output_flat.scatter_add_(1, global_indices_flat.unsqueeze(0).expand(C, -1), weighted_values)
        weight_sum.scatter_add_(0, global_indices_flat, all_w_flat)
        
        # Normalize
        output = output_flat / (weight_sum.unsqueeze(0) + 1e-8)
        output = output.view(C, B, H, W).permute(1, 0, 2, 3)
        
        # Cast back to original dtype
        if original_dtype == torch.float16:
            output = output.half()
        
        return output