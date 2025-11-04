import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- Supervised Loss ---

def EPE_loss(
    pred_disps: List[torch.Tensor],
    gt_disp: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.8
) -> torch.Tensor:
    """
    Computes the sequence-aware supervised loss for a list of disparity predictions.

    This loss is a weighted sum of the Smooth L1 loss calculated for each prediction
    in the sequence. The weights increase exponentially for later predictions, placing
    more importance on the final, more refined outputs.

    Args:
        pred_disps (List[torch.Tensor]): A list of predicted disparity maps, from
                                        initial estimates to final refinements.
                                        All should be upsampled to the full resolution.
        gt_disp (torch.Tensor): The ground truth disparity map. Shape: [B, H, W]
        mask (torch.Tensor): A boolean tensor indicating valid pixels in the
                             ground truth. Shape: [B, H, W]
        gamma (float): The exponential weight factor.

    Returns:
        torch.Tensor: The final computed scalar loss value.
    """
    # Compute Smooth L1 loss with GT disparity 
    loss_weights = [gamma ** (len(pred_disps) - 1 - power) for power in
                        range(len(pred_disps))]
    disp_loss = 0
    
    for k in range(len(pred_disps)):
        pred_disp = pred_disps[k]
        weight = loss_weights[k]

        curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask], reduction='mean')
        disp_loss += weight * curr_loss
        
    return disp_loss


# --- Regularization Loss ---

def edge_aware_smoothness_loss(
    pred_disps: List[torch.Tensor],
    image: torch.Tensor,
    gamma: float = 0.8
) -> torch.Tensor:
    """
    Computes the edge-aware smoothness loss.

    This loss encourages the predicted disparity to be locally smooth, while allowing
    for sharp changes at image edges. It penalizes the disparity gradient more heavily
    in flat image regions and less so in regions with strong image gradients.

    Args:
        pred_disp (torch.Tensor): The predicted disparity map. Shape: [B, 1, H, W]
        image (torch.Tensor): The corresponding input image (e.g., left image).
                              Shape: [B, 3, H, W]

    Returns:
        torch.Tensor: The computed smoothness loss.
    """
    loss_weights = [gamma ** (len(pred_disps) - 1 - power) for power in
                        range(len(pred_disps))]
    total_loss = 0

    for k in range(len(pred_disps)):
        pred_disp = pred_disps[k]
        weight = loss_weights[k]

        # Calculate disparity gradients
        disp_dy, disp_dx = torch.gradient(pred_disp, dim=(-2, -1))

        # Calculate image gradients (average over color channels)
        img_dy, img_dx = torch.gradient(torch.mean(image, 1, keepdim=True), dim=(-2, -1))

        # Calculate weights based on image gradients
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True))

        # Apply weights to disparity gradients
        smoothness_x = torch.mean(torch.abs(disp_dx) * weights_x)
        smoothness_y = torch.mean(torch.abs(disp_dy) * weights_y)

        curr_loss = smoothness_x + smoothness_y
        total_loss += weight * curr_loss

    return total_loss

def second_order_smoothness_loss(pred_disp: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """
    Computes the second-order smoothness loss (curvature penalty).

    This encourages piecewise-planar surfaces by penalizing the second derivative
    of the disparity map, weighted by the image gradients to preserve edges.

    Args:
        pred_disp (torch.Tensor): The predicted disparity map. Shape: [B, 1, H, W]
        image (torch.Tensor): The corresponding input image. Shape: [B, 3, H, W]

    Returns:
        torch.Tensor: The computed second-order smoothness loss.
    """
    # --- Calculate second-order disparity gradients (curvature) ---
    # We use padding to handle the boundaries
    paddings = (1, 1, 1, 1) # Pad left/right, top/bottom
    disp_padded = F.pad(pred_disp, paddings, mode="replicate")
    
    # d(x-1) - 2d(x) + d(x+1)
    disp_dxx = disp_padded[:, :, 1:-1, :-2] - 2 * pred_disp + disp_padded[:, :, 1:-1, 2:]
    # d(y-1) - 2d(y) + d(y+1)
    disp_dyy = disp_padded[:, :, :-2, 1:-1] - 2 * pred_disp + disp_padded[:, :, 2:, 1:-1]

    # --- Calculate image gradients to create edge-aware weights ---
    img_dy, img_dx = torch.gradient(torch.mean(image, 1, keepdim=True), dim=(-2, -1))
    
    # Weights should be high where the image is smooth, low where there are edges
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True))

    # --- Combine and compute the loss ---
    # We only need to slice the weights to match the size of the derivatives
    smoothness_x = torch.mean(torch.abs(disp_dxx) * weights_x[:, :, 1:-1, 1:-1])
    smoothness_y = torch.mean(torch.abs(disp_dyy) * weights_y[:, :, 1:-1, 1:-1])
    
    return smoothness_x + smoothness_y

# --- Self-Supervised Loss ---

def _ssim(x: torch.Tensor, y: torch.Tensor, C1: float = 1e-4, C2: float = 9e-4) -> torch.Tensor:
    """Helper function to compute the SSIM loss."""
    mu_x = F.avg_pool2d(x, 3, 1, padding=1)
    mu_y = F.avg_pool2d(y, 3, 1, padding=1)
    
    sigma_x = F.avg_pool2d(x**2, 3, 1, padding=1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1, padding=1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, padding=1) - mu_x * mu_y
    
    # Numerator
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    # Denominator
    ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    
    ssim = ssim_n / ssim_d
    
    return torch.clamp((1 - ssim) / 2, 0, 1)

def _reconstruct_image(disp: torch.Tensor, image_to_warp: torch.Tensor) -> torch.Tensor:
    """Helper to warp an image using a disparity map."""
    B, C, H, W = image_to_warp.shape
    
    # Create a base grid of pixel coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=disp.device, dtype=disp.dtype),
        torch.linspace(-1, 1, W, device=disp.device, dtype=disp.dtype),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]

    # Normalize disparity from pixels to [-1, 1] range for grid_sample
    # Note: Disparity is only applied to the x-coordinate
    normalized_disp = disp.squeeze(1) / ((W - 1) / 2.0)
    
    # Create the new sampling grid by shifting x-coordinates
    new_grid = grid.clone()
    new_grid[..., 0] = grid[..., 0] - normalized_disp

    # Sample from the right image to reconstruct the left
    return F.grid_sample(image_to_warp, new_grid, mode='bilinear', padding_mode='border', align_corners=True)

def photometric_reconstruction_loss(
    pred_disp: torch.Tensor,
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    ssim_weight: float = 0.85
) -> torch.Tensor:
    """
    Computes the self-supervised photometric reconstruction loss.

    This loss is used for training without ground truth. It works by warping the
    right image to match the left image using the predicted disparity. The difference
    between the original left image and the reconstructed one serves as the error signal.
    It combines an L1 loss with an SSIM loss.

    Args:
        pred_disp (torch.Tensor): The predicted left disparity map. Shape: [B, 1, H, W]
        left_img (torch.Tensor): The original left image. Shape: [B, 3, H, W]
        right_img (torch.Tensor): The original right image. Shape: [B, 3, H, W]
        ssim_weight (float): The weight for the SSIM component of the loss.

    Returns:
        torch.Tensor: The computed photometric loss.
    """
    # 1. Reconstruct the left image from the right image and disparity
    reconstructed_left = _reconstruct_image(pred_disp, right_img)

    # 2. Compute L1 loss
    l1_loss = torch.mean(torch.abs(reconstructed_left - left_img))
    
    # 3. Compute SSIM loss
    # ssim_loss = torch.mean(_ssim(reconstructed_left, left_img))

    # 4. Combine the two losses
    # total_loss = ssim_weight * ssim_loss + (1 - ssim_weight) * l1_loss
    
    # return total_loss
    return l1_loss


# --- Example Usage ---
if __name__ == '__main__':
    # --- Dummy Data Setup ---
    B, H, W = 2, 64, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Supervised learning data
    pred_disps_list = [
        torch.rand(B, 1, H, W, device=device) * 50,  # Initial coarse prediction
        torch.rand(B, 1, H, W, device=device) * 50,  # Refined prediction
        torch.rand(B, 1, H, W, device=device) * 50   # Final prediction
    ]
    ground_truth_disp = torch.rand(B, H, W, device=device) * 50
    valid_mask = torch.rand(B, H, W, device=device) > 0.2 # ~80% of pixels are valid

    # Self-supervised learning data
    left_image = torch.rand(B, 3, H, W, device=device)
    right_image = torch.rand(B, 3, H, W, device=device)
    final_pred_disp = pred_disps_list[-1]

    # --- Loss Calculation ---
    
    # 1. Supervised Sequence Loss
    sup_loss = sequence_loss(pred_disps_list, ground_truth_disp, valid_mask, gamma=0.85)
    print(f"Supervised Sequence Loss: {sup_loss.item():.4f}")

    # 2. Edge-Aware Smoothness Loss (a regularization term)
    smooth_loss = edge_aware_smoothness_loss(final_pred_disp, left_image)
    print(f"Edge-Aware Smoothness Loss: {smooth_loss.item():.4f}")

    # 3. Second-Order Smoothness Loss (a regularization term)
    smooth_loss = edge_aware_smoothness_loss(final_pred_disp, left_image)
    print(f"Edge-Aware Smoothness Loss: {smooth_loss.item():.4f}")

    # 4. Self-Supervised Photometric Loss
    photo_loss = photometric_reconstruction_loss(final_pred_disp, left_image, right_image, ssim_weight=0.85)
    print(f"Photometric Reconstruction Loss: {photo_loss.item():.4f}")
    
    # In a real training script, you might combine them, e.g.:
    # total_supervised_loss = sup_loss + 0.1 * smooth_loss
    # total_self_supervised_loss = photo_loss + 0.1 * smooth_loss
