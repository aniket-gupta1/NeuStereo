import torch.nn.functional as F

from NeuFlow import utils


class Matching:

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.grid = utils.coords_grid(batch_size, height, width, device, amp)  # [B, 2, H, W]
        self.flatten_grid = self.grid.view(batch_size, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    def global_correlation_softmax(self, feature0, feature1, correspondence_gt=None):

        b, c, h, w = feature0.shape

        feature0 = feature0.flatten(-2).permute(0, 2, 1)
        feature1 = feature1.flatten(-2).permute(0, 2, 1)

        correspondence = F.scaled_dot_product_attention(feature0, feature1, self.flatten_grid)
        correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # self.visualize_correspondence(correspondence, correspondence_gt)

        # raise ValueError

        flow = correspondence - self.grid

        return flow

    def visualize_correspondence(self, correspondence_pred, correspondence_gt, image_left=None, idx=0):
        """
        Visualize predicted vs GT correspondence maps.
        
        Args:
            correspondence_pred: [B, 2, H, W] - predicted correspondence
            correspondence_gt: [B, 2, H, W] - ground truth correspondence
            image_left: [B, 3, H, W] - optional left image for overlay
            idx: batch index to visualize
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract single sample
        pred = correspondence_pred[idx].cpu().numpy()  # [2, H, W]
        gt = correspondence_gt[idx].cpu().numpy()
        
        # Extract x-coordinates (most important for stereo)
        pred_x = pred[0]  # [H, W]
        gt_x = gt[0]
        
        # Compute error map
        error_x = np.abs(pred_x - gt_x)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # GT x-coordinate
        im1 = axes[0, 0].imshow(gt_x, cmap='jet')
        axes[0, 0].set_title('GT X-Correspondence')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Predicted x-coordinate  
        im2 = axes[0, 1].imshow(pred_x, cmap='jet')
        axes[0, 1].set_title('Predicted X-Correspondence')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Error map
        im3 = axes[0, 2].imshow(error_x, cmap='hot')
        axes[0, 2].set_title(f'Absolute Error (mean: {error_x.mean():.2f})')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Disparity maps (x_left - x_right)
        H, W = pred_x.shape
        x_left = np.arange(W)[None, :].repeat(H, axis=0)
        
        disp_gt = x_left - gt_x
        disp_pred = x_left - pred_x
        
        im4 = axes[1, 0].imshow(disp_gt, cmap='magma', vmin=0)
        axes[1, 0].set_title('GT Disparity')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(disp_pred, cmap='magma', vmin=0)
        axes[1, 1].set_title('Predicted Disparity')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Disparity error
        disp_error = np.abs(disp_pred - disp_gt)
        im6 = axes[1, 2].imshow(disp_error, cmap='hot')
        axes[1, 2].set_title(f'Disparity Error (mean: {disp_error.mean():.2f})')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f"comparison_correspondence.png")
        
        return fig
