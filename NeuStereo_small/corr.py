import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from NeuStereo import utils


def bilinear_sample(img, coords):
	""" Wrapper for grid_sample, uses pixel coordinates """
	H, W = img.shape[-2:]
	xgrid, ygrid = coords.split([1,1], dim=-1)
	xgrid = 2*xgrid/(W-1) - 1
	ygrid = 2*ygrid/(H-1) - 1

	grid = torch.cat([xgrid, ygrid], dim=-1)

	with torch.backends.cudnn.flags(enabled=False):
		img = F.grid_sample(img, grid, align_corners=True)

	return img


def bilinear_grid_sample(img: torch.Tensor,
                         coords: torch.Tensor,
                         mode: str = 'bilinear',
                         align_corners: bool = True) -> torch.Tensor:
    """
    A clean wrapper around `F.grid_sample` that takes pixel-space coords (x, y)
    and converts them to normalized [-1..1] space for sampling.

    Args:
      img:    [B, C, H, W]   (the image / volume to sample from)
      coords: [B, H_out, W_out, 2], where coords[...,0] = x, coords[...,1] = y
              in pixel coordinates (0-based).
      mode:   interpolation mode ('bilinear', 'nearest', etc.)
      align_corners: see PyTorch's grid_sample docs

    Returns:
      sampled: [B, C, H_out, W_out]
    """
    # Unpack shapes
    B, C, H, W = img.shape
    # coords => [B, H_out, W_out, 2]
    # split x and y
    xcoords, ycoords = coords[..., 0], coords[..., 1]  # each => [B, H_out, W_out]

    # Normalize x from [0..W-1] to [-1..1]
    # and y from [0..H-1] to [-1..1].
    xnorm = 2.0 * xcoords / (W - 1) - 1.0
    ynorm = 2.0 * ycoords / (H - 1) - 1.0

    # Recombine
    norm_grid = torch.stack([xnorm, ynorm], dim=-1)  # => [B, H_out, W_out, 2]

    sampled = F.grid_sample(img, norm_grid,
                            mode=mode,
                            padding_mode='zeros',
                            align_corners=align_corners)
    return sampled


class CorrBlock:
    def __init__(self, radius, levels):

        self.radius = radius
        self.levels = levels

    def init_bhwd(self, batch_size, height, width, device, amp):

        xy_range = torch.linspace(-self.radius, self.radius, 2*self.radius+1, dtype=torch.half if amp else torch.float, device=device)

        delta = torch.stack(torch.meshgrid(xy_range, xy_range, indexing='ij'), axis=-1)
        delta = delta.view(1, 2*self.radius+1, 2*self.radius+1, 2)

        self.grid = utils.coords_grid(batch_size, height, width, device, amp)
        self.delta = delta.repeat(batch_size * height * width, 1, 1, 1)

    def __call__(self, corr_pyramid, flow):

        b, _, h, w = flow.shape

        coords = (self.grid + flow).permute(0, 2, 3, 1)
        coords = coords.reshape(b*h*w, 1, 1, 2)

        out_list = []

        for level, corr in enumerate(corr_pyramid):
            curr_coords = coords / 2**level + self.delta
            corr = bilinear_sample(corr, curr_coords)
            corr = corr.view(b, h, w, -1)
            out_list.append(corr)

        out = torch.cat(out_list, dim=-1)

        return out.permute(0, 3, 1, 2).contiguous()

    def init_corr_pyr(self, feature0, feature1):
        b, c, h, w = feature0.shape
        feature0 = feature0.view(b, c, h*w)
        feature1 = feature1.view(b, c, h*w)
        
        corr = torch.matmul(feature0.transpose(1,2), feature1)
        corr = corr.view(b*h*w, 1, h, w) / math.sqrt(c)

        corr_pyramid = [corr]
        for i in range(self.levels-1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            corr_pyramid.append(corr)

        return corr_pyramid

class StereoCorrBlock:
    """
    A RAFT-like correlation block specialized for stereo.
    Computes row-by-row correlation, builds a multi-level pyramid,
    and performs local correlation lookups based on disparity.
    """
    def __init__(self, radius=4, levels=4):
        """
        Args:
          radius: local search radius (in pixels) around the disparity offset.
          levels: number of pyramid levels to build (each level is half res).
        """
        self.radius = radius
        self.levels = levels
        self.corr_pyramid = None  # Will hold [corr_level0, corr_level1, ...], each shape [B, H, W, W]
    
    def init_bhwd(self, B, H, W, device, amp):
        """
        Initialize the xL_base and offset values in here to prevent recomputing in every iteration
        """
        # 2.2) xL_base => row index in [0..W-1] 
        self.xL_base = torch.arange(W, device=device).view(1, W).float()  # => [1, W]
        self.xL_base = self.xL_base.expand(B*H, -1)  # => [B*H, W]

        # 2.3) The offsets in [-radius..radius]
        self.offset_vals = torch.arange(-self.radius, self.radius+1, device=device).view(1, 1, 2*self.radius+1).float() 
        self.offset_vals = self.offset_vals.expand(B*H, W, -1)  # => [B*H, W, 2*r+1]
        

    def build_stereo_corr_volume(self, feat_left, feat_right):
        """
        Compute the row-by-row correlation volume for a single scale.
        
        Args:
          feat_left, feat_right: shape [B, C, H, W]
        
        Returns:
          corr_volume: [B, H, W, W], 
            where corr_volume[b, i, xL, xR] = dot product of
            feat_left[b,:,i,xL] and feat_right[b,:,i,xR].
        """
        B, C, H, W = feat_left.shape

        # 1) Rearrange LEFT feature into [B*H, W, C]:
        #    - permute to [B, H, C, W]
        #    - reshape to [B*H, C, W]
        #    - then transpose last two dims => [B*H, W, C]
        left_ = feat_left.permute(0, 2, 1, 3)       # [B, H, C, W]
        left_ = left_.reshape(B * H, C, W)          # [B*H, C, W]
        left_ = left_.permute(0, 2, 1)              # [B*H, W, C]

        # 2) Rearrange RIGHT feature into [B*H, C, W]
        right_ = feat_right.permute(0, 2, 1, 3)     # [B, H, C, W]
        right_ = right_.reshape(B * H, C, W)        # [B*H, C, W]

        # 3) Batched matrix multiply => [B*H, W, W]
        #    For each row, we get a (W x W) correlation matrix
        corr = torch.bmm(left_, right_)  # [B*H, W, W]

        # 4) Reshape to [B, H, W, W] and scale
        corr_volume = corr.view(B, H, W, W) / math.sqrt(C)

        return corr_volume
    
    def local_correlation_lookup(self, corr_volume, disp, radius):
        """
        For each pixel, sample correlation values around the 'disparity' offset in the right dimension.
        
        Args:
          corr_volume: [B, H, W, W]
          disp:        [B, 1, H, W] (the predicted disparity at this scale)
          radius:      integer search radius
        Returns:
          cost_out: [B, (2*radius+1), H, W],
            stacked cost slices around disp offset.
        """
        B, H, W, _ = corr_volume.shape  # => [B, H, W, W]

        # 1) Flatten => treat each row as an image [B*H, 1, W, W]
        corr_flat = corr_volume.view(B*H, 1, W, W)  # => shape "batch"=B*H, single-channel, size=(W,W)

        # 2) We will sample multiple offsets at once: dimension => (2*r+1)
        # So we define "output grid" shape = [B*H, W, (2*r+1), 2], where 'W' is "height" and '2*r+1' is "width" in our grid-sample call.

        # 2.1) Flatten disp => [B*H, W]
        disp_2d = disp[:, 0].reshape(B*H, W)  # => [B*H, W]

        # 3) xR = xL_base - disp + offset
        #    yR = xL_base  (in the flattened view, "vertical"=xL, "horizontal"=xR)
        xR = self.xL_base.unsqueeze(-1) - disp_2d.unsqueeze(-1) + self.offset_vals  # => [B*H, W, 2*r+1]
        yR = self.xL_base.unsqueeze(-1).expand(-1, -1, 2*radius+1)             # => [B*H, W, 2*r+1]

        # 4) Stack => coords[...,0] = xR, coords[...,1] = yR
        # => [B*H, W, 2*r+1, 2]
        coords = torch.stack([xR, yR], dim=-1)

        # 5) bilinear sampling
        # We'll treat corr_flat as [B*H, 1, H'=W, W'=W].
        # Our grid => [B*H, H_out=W, W_out=(2*r+1), 2].
        sampled = bilinear_grid_sample(corr_flat, coords, mode='bilinear', align_corners=True)
        # => [B*H, 1, W, (2*r+1)]

        # 6) Rearrange => [B, (2*r+1), H, W]
        cost_out = sampled.view(B, H, 1, W, 2*radius+1)
        cost_out = cost_out.permute(0, 4, 1, 3, 2).squeeze(-1)
        # shape => [B, (2*r+1), H, W]

        return cost_out

    def init_corr_pyr(self, feat_left, feat_right):
        """
        Build a multi-level correlation pyramid by progressively downsampling features.
        
        Args:
          feat_left, feat_right: [B, C, H, W] at the highest resolution
        
        Returns:
          A list of correlation volumes [corr_lvl0, corr_lvl1, ...].
          Each corr_lvl has shape [B, H_l, W_l, W_l].
        """
        corr_pyramid = []
        
        current_left = feat_left
        current_right = feat_right

        for lvl in range(self.levels):
            corr_lvl = self.build_stereo_corr_volume(current_left, current_right)
            corr_pyramid.append(corr_lvl)

            # Downsample for the next level (except after the last level)
            if lvl < self.levels - 1:
                current_left  = F.avg_pool2d(current_left,  kernel_size=2, stride=2)
                current_right = F.avg_pool2d(current_right, kernel_size=2, stride=2)

        return corr_pyramid

    def __call__(self, corr_pyramid, disp):
        """
        For each pyramid level, do local correlation lookup around disparity, 
        then (optionally) upsample to full res if you want to concat them all.
        
        Args:
          disp: [B, 1, H, W] - the current disparity estimate at the highest resolution.
        
        Returns:
          A cost volume or list of cost volumes for each scale. 
          For demonstration, we'll create a single combined cost volume at full resolution
          by upsampling each level's local correlation.
        """
        
        B, _, H, W = disp.shape

        out_list = []
        for lvl, corr_lvl in enumerate(corr_pyramid):
            # corr_lvl: [B, H_l, W_l, W_l]
            scale_factor = 2 ** lvl
            H_l = H // scale_factor
            W_l = W // scale_factor

            # Downsample disparity to match corr_lvl resolution
            disp_l = F.interpolate(disp, size=(H_l, W_l), mode='bilinear', align_corners=False) / scale_factor

            # Local correlation => [B, (2*r+1), H_l, W_l]
            cost_l = self.local_correlation_lookup(corr_lvl, disp_l, self.radius)

            # (Optional) upsample cost_l back to full res 
            # so we can concat all levels. 
            cost_l_up = F.interpolate(cost_l, size=(H, W), mode='bilinear', align_corners=False)
            out_list.append(cost_l_up)

        # Concatenate along channel dim => shape [B, sum_of_channels, H, W]
        cost_final = torch.cat(out_list, dim=1)
        return cost_final
