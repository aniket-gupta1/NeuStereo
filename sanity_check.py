import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def unnormalize_image(image):
    # Check that the image has 3 dimensions and the shape - C, H, W
    assert image.ndim==3 and image.shape[0]==3, (image.shape, image.ndim)
    for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        # The reverse operation is to multiply by std and then add the mean
        t.mul_(s).add_(m)    
    return image

def _tensor_to_plot(tensor, image=False):
    """
    Converts a PyTorch tensor to a plottable NumPy array.
    Handles tensors on GPU, with batch dimension, and channel-first format.
    Ensures the final NumPy array has a dtype compatible with Matplotlib.
    """
    if tensor is None:
        return None

    # Select the first item in the batch
    if tensor.ndim == 4:
        tensor = tensor[0]

    if image:
        tensor = unnormalize_image(tensor)
        
    # Detach from graph, move to CPU, and convert to numpy
    # --- FIX: Explicitly cast to a supported dtype like float32 ---
    tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
        
    # Handle channel-first format (C, H, W)
    if tensor_np.ndim == 3 and tensor_np.shape[0] in [1, 3]:
        if tensor_np.shape[0] == 1:  # Grayscale/Disparity
            tensor_np = tensor_np.squeeze(0)  # Becomes (H, W)
        else:  # RGB Image
            tensor_np = np.transpose(tensor_np, (1, 2, 0))  # Becomes (H, W, C)
            
    # # Clip RGB images to [0, 1] range for correct display
    # if tensor_np.ndim == 3:
    #     tensor_np = np.clip(tensor_np, 0, 1)
        
    return tensor_np

def plot_video_stereo_debug(
    # Timestep 0 data
    left_img_t0, right_img_t0, gt_disp_t0, pred_disp_t0,
    # Timestep 1 data
    left_img_t1, right_img_t1, gt_disp_t1, pred_disp_t1,
    # Warped data for timestep 1
    warped_disp_t1,
    warped_features_t1,  # This is a dict {'s16': tensor, 's8': tensor}
    # Save configuration
    save_path="debug_plot.png"
):
    """
    Creates and saves a comprehensive debug plot for two consecutive timesteps
    of a video stereo model to verify the temporal warping process.
    """
    # pdb.set_trace()
    # Prepare all tensors for plotting using the helper function
    img_t0_l = _tensor_to_plot(left_img_t0, image=True)
    img_t0_r = _tensor_to_plot(right_img_t0, image=True)
    disp_t0_gt = _tensor_to_plot(gt_disp_t0)
    disp_t0_pred = _tensor_to_plot(pred_disp_t0)

    img_t1_l = _tensor_to_plot(left_img_t1, image=True)
    img_t1_r = _tensor_to_plot(right_img_t1, image=True)
    disp_t1_gt = _tensor_to_plot(gt_disp_t1)
    disp_t1_pred = _tensor_to_plot(pred_disp_t1)
    disp_t1_warped = _tensor_to_plot(warped_disp_t1)

    # Prepare warped features for plotting (we'll visualize the first channel of each)
    warped_feats_plots = {}
    if warped_features_t1:
        for key, feat_tensor in warped_features_t1.items():
            # Extract the first channel and convert it
            if feat_tensor is not None:
                warped_feats_plots[key] = _tensor_to_plot(feat_tensor[:, 0:1, :, :])

    # --- Create the plot layout ---
    num_feature_plots = len(warped_feats_plots)
    num_cols = max(5, 5 + num_feature_plots) # Accommodate the longest row
    num_rows = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4.2))
    
    # --- Plot Timestep 0 (Top Row) ---
    axes[0, 0].imshow(img_t0_l); axes[0, 0].set_title("t=0 Left Image")
    axes[0, 1].imshow(img_t0_r); axes[0, 1].set_title("t=0 Right Image")
    
    im_gt0 = axes[0, 2].imshow(disp_t0_gt, cmap='viridis'); axes[0, 2].set_title("t=0 GT Disparity")
    fig.colorbar(im_gt0, ax=axes[0, 2], orientation='horizontal', pad=0.1)

    im_pred0 = axes[0, 3].imshow(disp_t0_pred, cmap='viridis'); axes[0, 3].set_title("t=0 Predicted Disp")
    fig.colorbar(im_pred0, ax=axes[0, 3], orientation='horizontal', pad=0.1)

    # --- Plot Timestep 1 (Bottom Row) ---
    axes[1, 0].imshow(img_t1_l); axes[1, 0].set_title("t=1 Left Image")
    axes[1, 1].imshow(img_t1_r); axes[1, 1].set_title("t=1 Right Image")

    im_gt1 = axes[1, 2].imshow(disp_t1_gt, cmap='viridis'); axes[1, 2].set_title("t=1 GT Disparity")
    fig.colorbar(im_gt1, ax=axes[1, 2], orientation='horizontal', pad=0.1)

    im_pred1 = axes[1, 3].imshow(disp_t1_pred, cmap='viridis'); axes[1, 3].set_title("t=1 Predicted Disp")
    fig.colorbar(im_pred1, ax=axes[1, 3], orientation='horizontal', pad=0.1)

    im_warp_d = axes[1, 4].imshow(disp_t1_warped, cmap='viridis'); axes[1, 4].set_title("t=1 Warped Prev. Disp")
    fig.colorbar(im_warp_d, ax=axes[1, 4], orientation='horizontal', pad=0.1)

    # Plot warped features dynamically
    for i, (key, feat_plot) in enumerate(warped_feats_plots.items()):
        col_idx = 5 + i
        im_feat = axes[1, col_idx].imshow(feat_plot, cmap='plasma')
        axes[1, col_idx].set_title(f"t=1 Warped Feat '{key}' (Ch 0)")
        fig.colorbar(im_feat, ax=axes[1, col_idx], orientation='horizontal', pad=0.1)
        
    # --- Finalize and Save ---
    for ax in axes.ravel():
        ax.axis('off')
        
    plt.tight_layout(pad=2.0)
    
    # Ensure the directory for the save path exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Debug plot saved to {save_path}")

