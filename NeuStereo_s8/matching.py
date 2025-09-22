import torch.nn.functional as F
import torch

from NeuStereo import utils


class Matching:

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.grid = utils.coords_grid(batch_size, height, width, device, amp)  # [B, 2, H, W]
        self.flatten_grid = self.grid.view(batch_size, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

        x_coords_orig = torch.arange(width, device=device).view(1, width, 1).float()
        self.x_coords = x_coords_orig.expand(batch_size * height, -1, -1)  # [B*H, W, 1]

        self.left_x = torch.arange(width, device=device).view(1, width, 1).expand(batch_size * height, -1, -1).float() # [B*H, W, 1]

    def global_correlation_softmax(self, feature0, feature1):

        # b, c, h, w = feature0.shape

        # feature0 = feature0.flatten(-2).permute(0, 2, 1)
        # feature1 = feature1.flatten(-2).permute(0, 2, 1)

        # correspondence = F.scaled_dot_product_attention(feature0, feature1, self.flatten_grid)

        # correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # flow = correspondence - self.grid

        # return -flow[:, 0, :, :].unsqueeze(1)

        ## CODE FROM GMFLOW
        # global correlation on horizontal direction
        b, c, h, w = feature0.shape

        x_grid = torch.linspace(0, w - 1, w, device=feature0.device)  # [W]

        feature0 = feature0.permute(0, 2, 3, 1)  # [B, H, W, C]
        feature1 = feature1.permute(0, 2, 1, 3)  # [B, H, C, W]

        correlation = torch.matmul(feature0, feature1) / (c ** 0.5)  # [B, H, W, W]

        # mask subsequent positions to make disparity positive
        mask = torch.triu(torch.ones((w, w)), diagonal=1).type_as(feature0)  # [W, W]
        valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)  # [B, H, W, W]

        correlation[~valid_mask] = -1e4

        prob = F.softmax(correlation, dim=-1)  # [B, H, W, W]

        correspondence = (x_grid.view(1, 1, 1, w) * prob).sum(-1)  # [B, H, W]

        # NOTE: unlike flow, disparity is typically positive
        disparity = x_grid.view(1, 1, w).repeat(b, h, 1) - correspondence  # [B, H, W]

        return disparity.unsqueeze(1)

    def stereo_correlation_softmax(self, feature_left, feature_right):
        """
        Compute the disparity for stereo-rectified images using 1D correlation (attention)
        along each row.
        """
        B, C, H, W = feature_left.shape
        # x_coords = torch.arange(W, device=feature_left.device).view(1, W, 1).float()

        # Reshape so that each row is treated as a separate 'batch' for attention
        # [B, C, H, W] -> [B, H, C, W] -> [B*H, W, C]
        left_ = feature_left.permute(0, 2, 1, 3).reshape(B * H, W, C)
        right_ = feature_right.permute(0, 2, 1, 3).reshape(B * H, W, C)

        # If you're using scaled_dot_product_attention, Q and K each has shape [batch, seq_len, embed_dim]
        # Q = left_, K = right_, but you also need a 'value'.
        # For producing disparity, we typically want the index (x-position) in the right image that the left pixel attends to.
        # Suppose we have an array of x-coordinates you call `self.x_coords` of shape [1, W, 1], broadcasted appropriately.
        # Then in flattened form, it might be [B*H, W, 1]. 
        # You can pass that as the Value to the attention.

        # For instance:
        # x_coords = x_coords.expand(B * H, -1, -1)  # [B*H, W, 1]

        # scaled_dot_product_attention(Q=left_, K=right_, V=x_coords)
        # Output shape: [B*H, W, 1]
        # For each of the W 'query' positions in left_, we get a *weighted average* x-coordinate from right_.
        correspondences = F.scaled_dot_product_attention(left_, right_, self.x_coords)
        # shape: [B*H, W, 1]

        # Now disparity_map[i, xL, 0] is the *estimated x-position* in the right image that matches xL in the left.
        # The actual disparity is  xL - xR
        # We can subtract the query index from the attended index to get disparity:
        
        # Make an index grid for the left x-coords
        # left_x = torch.arange(W, device=feature_left.device).view(1, W, 1).expand(B*H, -1, -1).float()

        disparity_map = self.left_x - correspondences  # shape [B*H, W, 1]
        
        # Reshape back to [B, 1, H, W]
        disparity_map = disparity_map.view(B, H, W, 1).permute(0, 3, 1, 2)

        return disparity_map  # [B, 1, H, W]
