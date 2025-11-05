import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

# Import the official softsplat CUDA kernel
try:
    from external import softsplat
except ImportError:
    print("Error: 'softsplat' module not found.")
    print("Please install it from the authors' repository:")
    print("https://github.com/sniklaus/softmax-splatting")


@torch.jit.script
def _calculate_flow_jit_dynamic(
    disparity_scaled: torch.Tensor,     # (B, 1, H, W)
    relative_pose: torch.Tensor,        # (B, 4, 4)
    grid_uv: torch.Tensor,              # (B, 2, H, W) Pre-computed grid
    fx: torch.Tensor,                   # (B, 1, 1, 1)
    fy: torch.Tensor,                   # (B, 1, 1, 1)
    cx: torch.Tensor,                   # (B, 1, 1, 1)
    cy: torch.Tensor,                   # (B, 1, 1, 1)
    baseline: torch.Tensor,             # (B, 1, 1, 1)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (JIT function is unchanged from your version)
    """
    B, _, H, W = disparity_scaled.shape
    device = disparity_scaled.device

    # --- 1. Dynamic P_norm Calculation (Inside JIT) ---
    u_norm = (grid_uv[:, 0:1] - cx) / fx
    v_norm = (grid_uv[:, 1:2] - cy) / fy
    
    ones_buffer = torch.ones_like(u_norm)
    P_norm = torch.cat([u_norm, v_norm, ones_buffer], dim=1) # (B, 3, H, W)

    # --- 2. Unproject (Optimized) ---
    Z_t_minus_1 = (fx * baseline) / disparity_scaled.clamp(min=1e-6)
    P_t_minus_1 = Z_t_minus_1.expand(-1, 3, -1, -1) * P_norm

    # --- 3. To Homogeneous & Transform ---
    P_t_minus_1_flat = P_t_minus_1.view(B, 3, -1) # (B, 3, N)
    
    ones_bmm = torch.ones(B, 1, H * W, 
                          device=device, 
                          dtype=P_t_minus_1_flat.dtype)

    P_t_minus_1_hom = torch.cat([P_t_minus_1_flat, ones_bmm], dim=1) # (B, 4, N)
    P_t_hom = torch.bmm(relative_pose, P_t_minus_1_hom) # (B, 4, N)

    # --- 4. Re-project (Optimized) ---
    X_t = P_t_hom[:, 0:1] # (B, 1, N)
    Y_t = P_t_hom[:, 1:2] # (B, 1, N)
    Z_t = P_t_hom[:, 2:3].clamp(min=1e-6) # (B, 1, N)

    u_prime = (fx.view(B,1,1) * X_t / Z_t + cx.view(B,1,1))
    v_prime = (fy.view(B,1,1) * Y_t / Z_t + cy.view(B,1,1))

    coords_t = torch.cat([u_prime, v_prime], dim=1).view(B, 2, H, W)

    # --- 5. Calculate 2D flow field ---
    flow_field = coords_t - grid_uv
    importance = disparity_scaled # Return the raw disparity

    return flow_field, importance


class SoftmaxWarper(nn.Module):
    """
    Optimized implementation of the temporal warper from XR-Stereo
    for DYNAMIC, PER-BATCH intrinsics.
    
    --- OPTIMIZATION ---
    This version computes the flow field only ONCE at full resolution
    and then downsamples the flow field for other scales.
    """
    def __init__(self, full_shape: Tuple[int, int], 
                 scales: List[float] = [1.0, 0.125, 0.0625]):
        super(SoftmaxWarper, self).__init__()

        self.scales = scales
        H_full, W_full = full_shape
        
        # Scale factor for importance metric
        self.metric_scale = 0.02 

        for scale in scales:
            s_str = str(scale).replace(".", "_")
            H, W = int(H_full * scale), int(W_full * scale)

            grid_y, grid_x = torch.meshgrid(
                torch.arange(H), torch.arange(W), indexing="ij"
            )
            grid_uv = torch.stack([grid_x, grid_y], dim=0).half() 
            self.register_buffer(f"grid_uv_{s_str}", grid_uv)

    def forward(
        self,
        disparity_prev: torch.Tensor,
        context_prev_8: torch.Tensor,
        context_prev_16: torch.Tensor,
        relative_pose: torch.Tensor,
        intrinsics: torch.Tensor,     # (B, 4) tensor [fx, fy, cx, cy]
        baseline: torch.Tensor        # (B) tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Clamp disparity
        # disparity_prev = torch.clamp(disparity_prev, min=0.1)

        B = disparity_prev.shape[0]
        # Reshape intrinsics for broadcasting (4D for JIT)
        fx = intrinsics[:, 0].view(B, 1, 1, 1)
        fy = intrinsics[:, 1].view(B, 1, 1, 1)
        cx = intrinsics[:, 2].view(B, 1, 1, 1)
        cy = intrinsics[:, 3].view(B, 1, 1, 1)
        baseline_b = baseline.view(B, 1, 1, 1)

        # --- 1. Warp Full-Resolution (Scale 1.0) ---
        # --- THIS IS NOW CALLED ONLY ONCE ---
        # scale_1_0 = 1.0
        # s_str_1_0 = "1_0"
        
        # flow_1_0, importance_1_0 = _calculate_flow_jit_dynamic(
        #     disparity_prev,
        #     relative_pose,
        #     getattr(self, f"grid_uv_{s_str_1_0}").expand(B, -1, -1, -1),
        #     fx, fy, cx, cy, baseline_b
        # )
        # warped_disparity = softsplat.softsplat(
        #     tenIn=disparity_prev,
        #     tenFlow=flow_1_0,
        #     tenMetric=importance_1_0 * self.metric_scale,
        #     strMode='soft'
        # )

        # --- 2. Warp Context 1/8 (Scale 0.125) ---
        # --- OPTIMIZATION: Downsample flow and importance ---
        
        # Downsample flow. Must scale values by scale_factor.
        scale_0_125 = 0.125
        s_str_0_125 = "0_125"
        
        # Scale intrinsics
        fx_8, fy_8 = fx * scale_0_125, fy * scale_0_125
        cx_8, cy_8 = cx * scale_0_125, cy * scale_0_125

        disparity_prev_8 = F.interpolate(
            disparity_prev, scale_factor=scale_0_125, 
            mode='bilinear', align_corners=False
        )

        flow_0_125, importance_0_125 = _calculate_flow_jit_dynamic(
            disparity_prev_8,
            relative_pose,
            getattr(self, f"grid_uv_{s_str_0_125}").expand(B, -1, -1, -1),
            fx_8, fy_8, cx_8, cy_8, baseline_b
        )

        warped_context_8 = softsplat.softsplat(
            tenIn=context_prev_8,
            tenFlow=flow_0_125,
            tenMetric=importance_0_125 * self.metric_scale,
            strMode='soft'
        )
        
        # --- 3. Warp Context 1/16 (Scale 0.0625) ---
        # --- OPTIMIZATION: Downsample flow and importance ---
        scale_0_0625 = 0.5

        # Downsample flow
        flow_0_0625 = F.interpolate(
            flow_0_125, scale_factor=scale_0_0625,
            mode='bilinear', align_corners=False
        ) * scale_0_0625
        
        # Downsample importance metric
        importance_0_0625 = F.interpolate(
            importance_0_125, scale_factor=scale_0_0625,
            mode='bilinear', align_corners=False
        )

        warped_context_16 = softsplat.softsplat(
            tenIn=context_prev_16,
            tenFlow=flow_0_0625,
            tenMetric=importance_0_0625 * self.metric_scale,
            strMode='soft'
        )
        
        warped_contexts = {
            "s16": warped_context_16,
            's8': warped_context_8
        }

        return warped_contexts

