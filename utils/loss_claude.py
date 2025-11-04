import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StereoLosses:
    """Complete collection of loss functions for stereo disparity estimation."""
    
    def __init__(self, max_disp=192, device='cuda'):
        self.max_disp = max_disp
        self.device = device
        self.ssim = SSIM().to(device)
        
    # ============== Core Supervised Losses ==============
    
    def l1_loss(self, pred_disps, gt_disp, mask):
        """Multi-scale L1 loss as introduced in DispNet."""
        total_loss = 0
        for scale, pred in enumerate(pred_disps):
            diff = torch.abs(pred - gt_disp)
            loss = (diff * mask).sum() / mask.sum()
            # Weight higher resolution predictions more
            weight = 1.0 / (2 ** scale) if scale > 0 else 1.0
            total_loss += weight * loss
        return total_loss
    
    def smooth_l1_loss(self, pred_disps, gt_disp, mask, beta=1.0):
        """Smooth L1 (Huber) loss as used in GC-Net."""
        total_loss = 0
        for scale, pred in enumerate(pred_disps):
            diff = pred - gt_disp
            loss = torch.where(
                torch.abs(diff) < beta,
                0.5 * diff ** 2 / beta,
                torch.abs(diff) - 0.5 * beta
            )
            loss = (loss * mask).sum() / mask.sum()
            weight = 1.0 / (2 ** scale) if scale > 0 else 1.0
            total_loss += weight * loss
        return total_loss
    
    def soft_cross_entropy_loss(self, cost_volume, gt_disp, mask):
        """Soft cross-entropy loss as used in PSMNet.
        Args:
            cost_volume: BxDxHxW (D is disparity dimension)
            gt_disp: BxHxW
            mask: BxHxW
        """
        B, D, H, W = cost_volume.shape
        
        # Convert to probability distribution
        prob_volume = F.softmax(cost_volume, dim=1)
        
        # Create soft labels (Gaussian around GT)
        disp_values = torch.arange(0, D, device=cost_volume.device).float()
        disp_values = disp_values.view(1, D, 1, 1).expand(B, D, H, W)
        
        gt_disp_expanded = gt_disp.unsqueeze(1).expand(B, D, H, W)
        sigma = 1.0  # Gaussian std
        soft_labels = torch.exp(-0.5 * ((disp_values - gt_disp_expanded) / sigma) ** 2)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        
        # Cross entropy
        loss = -(soft_labels * torch.log(prob_volume + 1e-8)).sum(dim=1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    # ============== Self-Supervised Losses ==============
    
    def photometric_loss(self, pred_disp, left_img, right_img, alpha=0.85):
        """Photometric consistency loss (SSIM + L1) from Monodepth."""
        # Warp right image to left using disparity
        warped = self.warp_image(right_img, -pred_disp)
        
        # SSIM loss
        ssim_loss = self.ssim(left_img, warped).mean()
        
        # L1 loss
        l1_loss = torch.abs(left_img - warped).mean()
        
        # Combined loss
        loss = alpha * (1 - ssim_loss) + (1 - alpha) * l1_loss
        return loss
    
    def left_right_consistency_loss(self, left_disp, right_disp):
        """Left-right disparity consistency from Monodepth."""
        # Warp right disparity to left view
        warped_right_disp = self.warp_image(right_disp.unsqueeze(1), -left_disp).squeeze(1)
        
        # Consistency loss
        loss = torch.abs(left_disp - warped_right_disp).mean()
        return loss
    
    def edge_aware_smoothness_loss(self, disp, img):
        """Edge-aware smoothness loss from Monodepth."""
        # Compute image gradients
        grad_img_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(dim=1, keepdim=True)
        grad_img_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(dim=1, keepdim=True)
        
        # Compute disparity gradients
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        
        # Edge-aware weighting
        weight_x = torch.exp(-torch.mean(grad_img_x))
        weight_y = torch.exp(-torch.mean(grad_img_y))
        
        loss = (weight_x * grad_disp_x).mean() + (weight_y * grad_disp_y).mean()
        return loss
    
    # ============== Advanced Losses ==============
    
    def census_transform_loss(self, pred_disp, left_img, right_img, kernel_size=7):
        """Census transform loss for robustness to illumination changes."""
        # Apply census transform
        left_census = self.census_transform(left_img, kernel_size)
        right_census = self.census_transform(right_img, kernel_size)
        
        # Warp right census to left
        warped_right_census = self.warp_image(right_census, -pred_disp)
        
        # Hamming distance
        hamming = (left_census != warped_right_census).float().mean()
        return hamming
    
    def occlusion_aware_loss(self, left_disp, right_disp, left_img, right_img):
        """Occlusion-aware photometric loss."""
        # Forward-backward consistency check
        warped_right_disp = self.warp_image(right_disp.unsqueeze(1), -left_disp).squeeze(1)
        fb_consistency = torch.abs(left_disp - warped_right_disp)
        
        # Create occlusion mask (pixels with high inconsistency)
        occlusion_mask = (fb_consistency < 1.0).float()
        
        # Photometric loss only on non-occluded regions
        warped_right = self.warp_image(right_img, -left_disp)
        photo_loss = torch.abs(left_img - warped_right)
        
        loss = (photo_loss * occlusion_mask.unsqueeze(1)).sum() / (occlusion_mask.sum() + 1e-6)
        return loss
    
    def feature_metric_loss(self, pred_disp, left_img, right_img, feature_extractor=None):
        """Perceptual loss using deep features."""
        if feature_extractor is None:
            # Use ImageNet pre-trained VGG features
            return self.photometric_loss(pred_disp, left_img, right_img)
        
        # Extract features
        left_feats = feature_extractor(left_img)
        right_feats = feature_extractor(right_img)
        
        # Warp features
        warped_feats = self.warp_image(right_feats, -pred_disp.unsqueeze(1).repeat(1, right_feats.shape[1], 1, 1))
        
        # Feature loss
        loss = torch.abs(left_feats - warped_feats).mean()
        return loss
    
    def second_order_smoothness_loss(self, disp):
        """Second-order smoothness from RAFT-Stereo for planar surfaces."""
        # Second derivatives
        ddx = disp[:, :, :, 2:] - 2 * disp[:, :, :, 1:-1] + disp[:, :, :, :-2]
        ddy = disp[:, :, 2:, :] - 2 * disp[:, :, 1:-1, :] + disp[:, :, :-2, :]
        
        loss = torch.abs(ddx).mean() + torch.abs(ddy).mean()
        return loss
    
    # ============== 3D Geometric Losses ==============
    
    def geometric_consistency_loss(self, disp, K, baseline=0.1):
        """3D geometric consistency loss from CODD.
        Args:
            disp: BxHxW disparity map
            K: Bx3x3 intrinsic matrix
            baseline: stereo baseline in meters
        """
        B, H, W = disp.shape
        
        # Convert disparity to depth
        fx = K[:, 0, 0].view(B, 1, 1)
        depth = (fx * baseline) / (disp + 1e-6)
        
        # Create pixel grid
        xx, yy = torch.meshgrid(torch.arange(W, device=disp.device), 
                                torch.arange(H, device=disp.device), indexing='xy')
        xx = xx.float().unsqueeze(0).expand(B, -1, -1)
        yy = yy.float().unsqueeze(0).expand(B, -1, -1)
        
        # Backproject to 3D
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        
        X = (xx - cx) * depth / fx
        Y = (yy - cy) * depth / fy
        Z = depth
        
        # Compute 3D point consistency (neighbor distances should be smooth)
        dx = torch.sqrt((X[:, :, 1:] - X[:, :, :-1])**2 + 
                       (Y[:, :, 1:] - Y[:, :, :-1])**2 + 
                       (Z[:, :, 1:] - Z[:, :, :-1])**2)
        dy = torch.sqrt((X[:, 1:, :] - X[:, :-1, :])**2 + 
                       (Y[:, 1:, :] - Y[:, :-1, :])**2 + 
                       (Z[:, 1:, :] - Z[:, :-1, :])**2)
        
        # Variance of neighbor distances (should be low for smooth surfaces)
        loss = dx.var() + dy.var()
        return loss
    
    def normal_consistency_loss(self, disp, K, baseline=0.1):
        """Surface normal consistency loss."""
        B, H, W = disp.shape
        
        # Convert to depth
        fx = K[:, 0, 0].view(B, 1, 1)
        depth = (fx * baseline) / (disp + 1e-6)
        
        # Compute normals from depth gradients
        dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        
        # Pad to original size
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        
        # Normal vector (simplified)
        normal = torch.stack([-dx, -dy, torch.ones_like(dx)], dim=1)
        normal = F.normalize(normal, dim=1)
        
        # Smoothness of normal field
        dn_dx = torch.abs(normal[:, :, :, 1:] - normal[:, :, :, :-1]).sum(dim=1)
        dn_dy = torch.abs(normal[:, :, 1:, :] - normal[:, :, :-1, :]).sum(dim=1)
        
        loss = dn_dx.mean() + dn_dy.mean()
        return loss
    
    # ============== Helper Functions ==============
    
    def warp_image(self, img, disp):
        """Warp image using disparity."""
        B, C, H, W = img.shape
        
        # Create mesh grid
        xx, yy = torch.meshgrid(torch.arange(W, device=img.device), 
                                torch.arange(H, device=img.device), indexing='xy')
        xx = xx.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        yy = yy.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        # Apply disparity
        xx_warped = xx + disp.unsqueeze(1)
        
        # Normalize to [-1, 1]
        xx_warped = 2.0 * xx_warped / (W - 1) - 1.0
        yy_warped = 2.0 * yy / (H - 1) - 1.0
        
        grid = torch.cat([xx_warped, yy_warped], dim=1).permute(0, 2, 3, 1)
        
        # Sample
        warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped
    
    def census_transform(self, img, kernel_size=7):
        """Apply census transform to image."""
        B, C, H, W = img.shape
        k = kernel_size // 2
        
        # Convert to grayscale if needed
        if C > 1:
            img = img.mean(dim=1, keepdim=True)
        
        # Pad image
        img_pad = F.pad(img, (k, k, k, k), mode='replicate')
        
        # Census transform
        census = torch.zeros(B, kernel_size*kernel_size, H, W, device=img.device)
        center = img
        
        idx = 0
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                if i == 0 and j == 0:
                    continue
                shifted = img_pad[:, :, k+i:H+k+i, k+j:W+k+j]
                census[:, idx] = (shifted > center).float().squeeze(1)
                idx += 1
        
        return census
    
    # ============== Combined Loss Function ==============
    
    def combined_supervised_loss(self, pred_disps, gt_disp, mask, weights=None):
        """Combine multiple supervised losses with weights."""
        if weights is None:
            weights = {
                'l1': 1.0,
                'smooth_l1': 0.5,
                'second_order': 0.1
            }
        
        total_loss = 0
        losses = {}
        
        if 'l1' in weights:
            l1 = self.l1_loss(pred_disps, gt_disp, mask)
            total_loss += weights['l1'] * l1
            losses['l1'] = l1
        
        if 'smooth_l1' in weights:
            smooth = self.smooth_l1_loss(pred_disps, gt_disp, mask)
            total_loss += weights['smooth_l1'] * smooth
            losses['smooth_l1'] = smooth
        
        if 'second_order' in weights and len(pred_disps) > 0:
            second = self.second_order_smoothness_loss(pred_disps[0])
            total_loss += weights['second_order'] * second
            losses['second_order'] = second
        
        return total_loss, losses
    
    def combined_self_supervised_loss(self, pred_disp, left_img, right_img, 
                                     right_disp=None, weights=None):
        """Combine multiple self-supervised losses."""
        if weights is None:
            weights = {
                'photometric': 1.0,
                'smoothness': 0.01,
                'lr_consistency': 1.0 if right_disp is not None else 0.0,
                'second_order': 0.01
            }
        
        total_loss = 0
        losses = {}
        
        if 'photometric' in weights:
            photo = self.photometric_loss(pred_disp, left_img, right_img)
            total_loss += weights['photometric'] * photo
            losses['photometric'] = photo
        
        if 'smoothness' in weights:
            smooth = self.edge_aware_smoothness_loss(pred_disp, left_img)
            total_loss += weights['smoothness'] * smooth
            losses['smoothness'] = smooth
        
        if 'lr_consistency' in weights and right_disp is not None:
            lr = self.left_right_consistency_loss(pred_disp, right_disp)
            total_loss += weights['lr_consistency'] * lr
            losses['lr_consistency'] = lr
        
        if 'second_order' in weights:
            second = self.second_order_smoothness_loss(pred_disp)
            total_loss += weights['second_order'] * second
            losses['second_order'] = second
        
        return total_loss, losses


class SSIM(nn.Module):
    """SSIM loss module."""
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)
        
    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                                 for x in range(window_size)])
            return gauss/gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map


# Example usage
def train_step_example(model, left_img, right_img, gt_disp, mask, optimizer, supervised=True):
    """Example training step using the loss functions."""
    loss_module = StereoLosses(max_disp=192)
    
    # Get predictions at multiple scales
    pred_disps = model(left_img, right_img)  # List of predictions
    
    if supervised:
        # Supervised training
        loss, loss_dict = loss_module.combined_supervised_loss(
            pred_disps, gt_disp, mask,
            weights={'l1': 1.0, 'smooth_l1': 0.5, 'second_order': 0.1}
        )
    else:
        # Self-supervised training
        # Assume model also predicts right disparity for consistency
        right_disp = model(right_img, left_img)[0]  
        
        loss, loss_dict = loss_module.combined_self_supervised_loss(
            pred_disps[0], left_img, right_img, right_disp,
            weights={'photometric': 1.0, 'smoothness': 0.01, 'lr_consistency': 1.0}
        )
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, loss_dict