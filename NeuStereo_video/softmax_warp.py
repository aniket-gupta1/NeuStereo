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
    fx: torch.Tensor,                   # (B, 1, 1, 1) <-- FIX: Must be 4D
    fy: torch.Tensor,                   # (B, 1, 1, 1) <-- FIX: Must be 4D
    cx: torch.Tensor,                   # (B, 1, 1, 1) <-- FIX: Must be 4D
    cy: torch.Tensor,                   # (B, 1, 1, 1) <-- FIX: Must be 4D
    baseline: torch.Tensor,             # (B, 1, 1, 1) <-- FIX: Must be 4D
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled function for 3D-to-2D flow with DYNAMIC intrinsics.
    All scalar inputs (fx, fy, cx, cy, baseline) must be 4D: [B, 1, 1, 1].
    """
    B, _, H, W = disparity_scaled.shape
    device = disparity_scaled.device

    # --- 1. Dynamic P_norm Calculation (Inside JIT) ---
    # (B,1,H,W) - (B,1,1,1) -> (B,1,H,W) (This broadcast is now unambiguous)
    u_norm = (grid_uv[:, 0:1] - cx) / fx
    v_norm = (grid_uv[:, 1:2] - cy) / fy
    
    ones_buffer = torch.ones_like(u_norm)
    P_norm = torch.cat([u_norm, v_norm, ones_buffer], dim=1) # (B, 3, H, W)

    # --- 2. Unproject (Optimized) ---
    # (B,1,1,1) * (B,1,1,1) / (B,1,H,W) -> (B,1,H,W)
    Z_t_minus_1 = (fx * baseline) / disparity_scaled.clamp(min=1e-6)
    
    # --- ERROR WAS HERE ---
    # (B,1,H,W) * (B,3,H,W)
    # This explicit broadcast makes the JIT compiler's job easy.
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

    # (B,1,1,1) * (B,1,N) / (B,1,N) + (B,1,1,1)
    # We must .view() the 4D intrinsics back to 3D for bmm-style broadcast
    u_prime = (fx.view(B,1,1) * X_t / Z_t + cx.view(B,1,1))
    v_prime = (fy.view(B,1,1) * Y_t / Z_t + cy.view(B,1,1))

    coords_t = torch.cat([u_prime, v_prime], dim=1).view(B, 2, H, W)

    # --- 5. Calculate 2D flow field ---
    flow_field = coords_t - grid_uv
    importance = disparity_scaled

    return flow_field, importance


class SoftmaxWarper(nn.Module):
    """
    Optimized implementation of the temporal warper from XR-Stereo
    for DYNAMIC, PER-BATCH intrinsics.

    This module pre-computes the static (u,v) grid and uses a
    JIT-compiled function to handle all dynamic calculations.
    
    Args:
        full_shape (Tuple[int, int]): The full (H, W) of the input.
        scales (List[float]): List of scales to operate on 
                             (e.g., [1.0, 0.125, 0.0625]).
    """
    def __init__(self, full_shape: Tuple[int, int], 
                 scales: List[float] = [1.0, 0.125, 0.0625]):
        super(SoftmaxWarper, self).__init__()

        self.scales = scales
        H_full, W_full = full_shape

        for scale in scales:
            s_str = str(scale).replace(".", "_")
            H, W = int(H_full * scale), int(W_full * scale)

            # Create static coordinate grid (u, v)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H), torch.arange(W), indexing="ij"
            )
            grid_uv = torch.stack([grid_x, grid_y], dim=0).half() # (2, H, W)
            # Register as buffer (will be moved to device with model)
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
        
        # # Clamp disparity to a small positive number to prevent
        # # numerical instability (e.g., division by zero, log(neg)).
        # disparity_prev = torch.clamp(disparity_prev, min=0.1)
        assert disparity_prev.min()>0.0

        # print("disparity shape: ", disparity_prev.shape)
        # print("disparity min: ", disparity_prev.min())
        # print("disparity max: ", disparity_prev.max())
        # print("context_prev_8: ", context_prev_8.shape)
        # print("context_prev_16: ", context_prev_16.shape)
        # print("relative_pose: ", relative_pose.shape)
        # print("intrinsics: ", intrinsics.shape)
        # print("baselines: ", baseline.shape)

        B = disparity_prev.shape[0]
        # Reshape intrinsics for broadcasting
        fx = intrinsics[:, 0].view(B, 1, 1, 1)
        fy = intrinsics[:, 1].view(B, 1, 1, 1)
        cx = intrinsics[:, 2].view(B, 1, 1, 1)
        cy = intrinsics[:, 3].view(B, 1, 1, 1)
        baseline_b = baseline.view(B, 1, 1, 1)

        # --- 1. Warp Full-Resolution Disparity (Scale 1.0) ---
        scale_1_0 = 1.0
        s_str_1_0 = "1_0"
        
        flow_1_0, importance_1_0 = _calculate_flow_jit_dynamic(
            disparity_prev,
            relative_pose,
            getattr(self, f"grid_uv_{s_str_1_0}").expand(B, -1, -1, -1),
            fx, fy, cx, cy, baseline_b
        )
        warped_disparity = softsplat.softsplat(
            tenIn=disparity_prev,
            tenFlow=flow_1_0,
            tenMetric=(-20.0 * importance_1_0).clip(-20.0, 20.0),
            strMode='soft'
        )

        # --- 2. Warp Context 1/8 (Scale 0.125) ---
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
            tenMetric=(-20.0 * importance_0_125).clip(-20.0, 20.0),
            strMode='soft'
        )
        
        # --- 3. Warp Context 1/16 (Scale 0.0625) ---
        scale_0_0625 = 0.0625
        s_str_0_0625 = "0_0625"

        # Scale intrinsics
        fx_16, fy_16 = fx * scale_0_0625, fy * scale_0_0625
        cx_16, cy_16 = cx * scale_0_0625, cy * scale_0_0625
        
        disparity_prev_16 = F.interpolate(
            disparity_prev, scale_factor=scale_0_0625, 
            mode='bilinear', align_corners=False
        )

        flow_0_0625, importance_0_0625 = _calculate_flow_jit_dynamic(
            disparity_prev_16,
            relative_pose,
            getattr(self, f"grid_uv_{s_str_0_0625}").expand(B, -1, -1, -1),
            fx_16, fy_16, cx_16, cy_16, baseline_b
        )
        warped_context_16 = softsplat.softsplat(
            tenIn=context_prev_16,
            tenFlow=flow_0_0625,
            tenMetric=(-20.0 * importance_0_0625).clip(-20.0, 20.0), 
            strMode='soft'
        )
        
        warped_contexts = {
            "s16": warped_context_16,
            's8': warped_context_8
        }

        return warped_contexts, warped_disparity

# --- Example Usage (Demonstration) ---
if __name__ == "__main__":
    # This example will only run if you have the 'softsplat' module
    # and a GPU (CUDA) available.
    B, C1, C2 = 2, 64, 128
    H, W = 480, 640

    # 1. Define shape and intrinsics for the module
    full_shape = (H, W)
    intrinsics = {
        'fx': 320.0, 'fy': 320.0,
        'cx': 319.5, 'cy': 239.5,
        'baseline': 0.1
    }
    
    # 2. Initialize the warper
    warper = SoftmaxWarper(full_shape, intrinsics)
    print("Initialized optimized SoftmaxWarper.")
    print(f"Trainable alpha: {warper.alpha.item()}")

    # 3. Create dummy inputs (on GPU)
    device = torch.device("cuda")
    warper.to(device)

    disparity_prev = (torch.rand(B, 1, H, W) * 50 + 10).to(device)
    context_prev_8 = torch.rand(B, C1, H // 8, W // 8).to(device)
    context_prev_16 = torch.rand(B, C2, H // 16, W // 16).to(device)
    
    relative_pose = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)
    relative_pose[:, 0, 3] = 0.1 # 10cm translation in X

    # 4. Run the forward pass
    print("Running forward pass...")
    with torch.no_grad():
        warped_disparity, warped_context_8, warped_context_16 = warper(
            disparity_prev,
            context_prev_8,
            context_prev_16,
            relative_pose
        )
    print("...Forward pass finished.")

    # 5. Check output shapes
    print(f"\n--- Output Shapes ---")
    print(f"Warped Disparity:  {warped_disparity.shape}")
    print(f"Warped Context 1/8:  {warped_context_8.shape}")
    print(f"Warped Context 1/16: {warped_context_16.shape}")

