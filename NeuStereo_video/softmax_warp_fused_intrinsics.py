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
def _calculate_flow_jit(
    disparity_scaled: torch.Tensor,
    relative_pose: torch.Tensor,
    grid_uv: torch.Tensor,
    P_norm: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> torch.Tensor:
    """
    JIT-compiled function to perform the 3D-to-2D flow calculation.
    
    This function is optimized to run on the GPU and fuses many
    element-wise operations.
    
    Args:
        disparity_scaled: (B, 1, H, W)
        relative_pose: (B, 4, 4)
        grid_uv: (B, 2, H, W) Pre-computed coordinate grid
        P_norm: (B, 3, H, W) Pre-computed normalized coordinates
        fx, fy, cx, cy, baseline: Scaled intrinsic parameters

    Returns:
        flow_field: (B, 2, H, W)
    """
    B, _, H, W = disparity_scaled.shape
    device = disparity_scaled.device

    # --- 1. Unproject (Optimized) ---
    # Z = f * b / d
    # P_t-1 = Z * P_norm
    Z_t_minus_1 = (fx * baseline) / disparity_scaled.clamp(min=1e-6)
    P_t_minus_1 = Z_t_minus_1 * P_norm  # (B, 3, H, W)

    # --- 2. To Homogeneous & Transform ---
    # Reshape for bmm: (B, 3, N) -> (B, 4, N)
    P_t_minus_1_hom = torch.cat(
        [
            P_t_minus_1.view(B, 3, -1),
            torch.ones(B, 1, H * W, device=device),
        ],
        dim=1,
    )
    
    # P_t = [R|T] * P_t-1
    # (B, 4, 4) @ (B, 4, N) -> (B, 4, N)
    P_t_hom = torch.bmm(relative_pose, P_t_minus_1_hom)

    # --- 3. Re-project (Optimized) ---
    X_t = P_t_hom[:, 0:1]  # (B, 1, N)
    Y_t = P_t_hom[:, 1:2]  # (B, 1, N)
    Z_t = P_t_hom[:, 2:3].clamp(min=1e-6)  # (B, 1, N)

    # u' = fx * X' / Z' + cx
    # v' = fy * Y' / Z' + cy
    u_prime = (fx * X_t / Z_t + cx)
    v_prime = (fy * Y_t / Z_t + cy)

    # Reshape to image: (B, 2, N) -> (B, 2, H, W)
    coords_t = torch.cat([u_prime, v_prime], dim=1).view(B, 2, H, W)

    # --- 4. Calculate 2D flow field ---
    # Flow = (u', v') - (u, v)
    flow_field = coords_t - grid_uv

    return flow_field


class SoftmaxWarper(nn.Module):
    """
    Optimized implementation of the temporal warper from XR-Stereo,
    using the official 'softsplat' CUDA kernel.

    Args:
        full_shape (Tuple[int, int]): The full (H, W) of the input.
        intrinsics (Dict): Camera intrinsics (fx, fy, cx, cy, baseline).
        scales (List[float]): List of scales to operate on.
        alpha (float): Initial value for the trainable 'alpha' 
                       (importance scaling) parameter.
    """
    def __init__(self, full_shape: Tuple[int, int], 
                 intrinsics: Dict[str, float], 
                 scales: List[float] = [1.0, 0.125, 0.0625],
                 alpha: float = 20.0):
        super(SoftmaxWarper, self).__init__()

        self.scales = scales
        self.intrinsics_scaled = {}
        self.baseline = intrinsics["baseline"] # Baseline is constant

        # --- Add trainable alpha parameter ---
        # This scales the importance metric (disparity), as shown in
        # the softsplat example code. It can be learned.
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))

        H_full, W_full = full_shape
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        for scale in scales:
            s_str = str(scale).replace(".", "_") # for buffer name
            
            # 1. Calculate scaled shape and intrinsics
            H, W = int(H_full * scale), int(W_full * scale)
            fx_s, fy_s = fx * scale, fy * scale
            cx_s, cy_s = cx * scale, cy * scale
            
            self.intrinsics_scaled[scale] = {
                "fx": fx_s, "fy": fy_s, "cx": cx_s, "cy": cy_s
            }

            # 2. Create static coordinate grid (u, v)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H), torch.arange(W), indexing="ij"
            )
            grid_uv = torch.stack([grid_x, grid_y], dim=0).float() # (2, H, W)
            self.register_buffer(f"grid_uv_{s_str}", grid_uv)

            # 3. Create static normalized coordinates (u_norm, v_norm, 1)
            u_norm = (grid_uv[0] - cx_s) / fx_s
            v_norm = (grid_uv[1] - cy_s) / fy_s
            P_norm = torch.stack(
                [u_norm, v_norm, torch.ones_like(u_norm)], dim=0
            ) # (3, H, W)
            self.register_buffer(f"P_norm_{s_str}", P_norm)

    def forward(
        self,
        disparity_prev: torch.Tensor,
        context_prev_8: torch.Tensor,
        context_prev_16: torch.Tensor,
        relative_pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B = disparity_prev.shape[0]

        # --- 1. Warp Full-Resolution Disparity (Scale 1.0) ---
        scale_1_0 = 1.0
        s_str_1_0 = "1_0"
        intrinsics_1_0 = self.intrinsics_scaled[scale_1_0]
        
        flow_1_0 = _calculate_flow_jit(
            disparity_prev,
            relative_pose,
            getattr(self, f"grid_uv_{s_str_1_0}").expand(B, -1, -1, -1),
            getattr(self, f"P_norm_{s_str_1_0}").expand(B, -1, -1, -1),
            intrinsics_1_0["fx"], intrinsics_1_0["fy"],
            intrinsics_1_0["cx"], intrinsics_1_0["cy"], self.baseline
        )
        
        # Scale and clip the metric, as per the softsplat example
        metric_1_0 = (self.alpha * disparity_prev).clip(-20.0, 20.0)
        
        warped_disparity = softsplat.softsplat(
            tenIn=disparity_prev,
            tenFlow=flow_1_0,
            tenMetric=metric_1_0,
            strMode='soft'
        )

        # --- 2. Warp Context 1/8 (Scale 0.125) ---
        scale_0_125 = 0.125
        s_str_0_125 = "0_125"
        intrinsics_0_125 = self.intrinsics_scaled[scale_0_125]
        
        disparity_prev_8 = F.interpolate(
            disparity_prev, scale_factor=scale_0_125, 
            mode='bilinear', align_corners=False
        )

        flow_0_125 = _calculate_flow_jit(
            disparity_prev_8,
            relative_pose,
            getattr(self, f"grid_uv_{s_str_0_125}").expand(B, -1, -1, -1),
            getattr(self, f"P_norm_{s_str_0_125}").expand(B, -1, -1, -1),
            intrinsics_0_125["fx"], intrinsics_0_125["fy"],
            intrinsics_0_125["cx"], intrinsics_0_125["cy"], self.baseline
        )
        
        metric_0_125 = (self.alpha * disparity_prev_8).clip(-20.0, 20.0)
        
        warped_context_8 = softsplat.softsplat(
            tenIn=context_prev_8,
            tenFlow=flow_0_125,
            tenMetric=metric_0_125,
            strMode='soft'
        )
        
        # --- 3. Warp Context 1/16 (Scale 0.0625) ---
        scale_0_0625 = 0.0625
        s_str_0_0625 = "0_0625"
        intrinsics_0_0625 = self.intrinsics_scaled[scale_0_0625]

        disparity_prev_16 = F.interpolate(
            disparity_prev, scale_factor=scale_0_0625, 
            mode='bilinear', align_corners=False
        )

        flow_0_0625 = _calculate_flow_jit(
            disparity_prev_16,
            relative_pose,
            getattr(self, f"grid_uv_{s_str_0_0625}").expand(B, -1, -1, -1),
            getattr(self, f"P_norm_{s_str_0_0625}").expand(B, -1, -1, -1),
            intrinsics_0_0625["fx"], intrinsics_0_0625["fy"],
            intrinsics_0_0625["cx"], intrinsics_0_0625["cy"], self.baseline
        )
        
        metric_0_0625 = (self.alpha * disparity_prev_16).clip(-20.0, 20.0)
        
        warped_context_16 = softsplat.softsplat(
            tenIn=context_prev_16,
            tenFlow=flow_0_0625,
            tenMetric=metric_0_0625,
            strMode='soft'
        )
        
        return warped_disparity, warped_context_8, warped_context_16

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

