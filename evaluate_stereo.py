import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any
# from datasets import build_dataset
from dataloader.datasets import (FlyingThings3D, KITTI15, ETH3DStereo, MiddleburyEval3)
from dataloader import transforms
from utils.utils import InputPadder
from utils.stereo_metric import epe_metric, d1_metric, thres_metric
import pdb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

# --- Constants ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Helper Functions ---
def count_parameters(model):
    actual_model = model.module if hasattr(model, 'module') else model
    return sum(p.numel() for p in actual_model.parameters() if p.requires_grad)

def benchmark_model(model, device):
    elapsed_list = []
    print("Working on image size: (512, 384)")
    for i in range(50):
        image1, image2 = torch.rand(3, 384, 512), torch.rand(3, 384, 512)
        image1 = image1[None].to(device)
        image1 = image1[None].to(device).half()
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=16)
        image1, image2 = padder.pad(image1, image2)
        
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

        # Benchmark using torch.utils.benchmark
        timer = benchmark.Timer(
            stmt=""" 
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):  
                results = model(image1, image2)
            """,
            globals={"model": model, "image1": image1, "image2": image2},
            num_threads=torch.get_num_threads(),
        )

        # Run timing (N=10 gives better stability)
        time_taken = timer.timeit(10).mean * 1000  # Convert to milliseconds
        elapsed_list.append(time_taken)

    return np.mean(elapsed_list)

def unnormalize_image(image):
    # Check that the image has 3 dimensions and the shape - C, H, W
    assert image.ndim==3 and image.shape[0]==3, (image.shape, image.ndim)
    for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        # The reverse operation is to multiply by std and then add the mean
        t.mul_(s).add_(m)
    return image

def save_outputs_func(
    image1: torch.Tensor, image2: torch.Tensor, disp_gt: torch.Tensor, 
    disp_pred: torch.Tensor, val_id: int, folder_name: str
):
    """Saves visual comparison of ground truth and predicted disparity."""

    # Make the output folder
    os.makedirs(f"outputs/{folder_name}", exist_ok=True)

    # Convert tensors to NumPy and normalize if needed
    image1 = unnormalize_image(image1)
    image2 = unnormalize_image(image2)
    img1_np = image1.permute(1, 2, 0).numpy()
    img2_np = image2.permute(1, 2, 0).numpy()
    disp_gt_np = disp_gt.cpu().numpy()
    pred_disp_np = disp_pred.cpu().numpy()
    
    # Make the valid mask
    valid_gt_mask = disp_gt_np > 0

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))  # Create a 2x2 grid
   
    # Display images
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title("Left Image")
    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title("Right Image")

    # GT Disparity with colorbar
    im3 = axes[1, 0].imshow(disp_gt_np, cmap="viridis")
    axes[1, 0].set_title(f"Ground Truth Disp. (min: {disp_gt_np.min():.2f}, max: {disp_gt_np.max():.2f})")
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label('Disparity (pixels)', rotation=270, labelpad=15)

    # Predicted Disparity with colorbar
    im4 = axes[1, 1].imshow(pred_disp_np, cmap="viridis")
    axes[1, 1].set_title(f"Predicted Disp. (min: {pred_disp_np.min():.2f}, max: {pred_disp_np.max():.2f})")
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar4.set_label('Disparity (pixels)', rotation=270, labelpad=15)

    # Error map with colorbar
    # Calculate error
    error_map = np.abs(pred_disp_np - disp_gt_np)
    error_map = np.where(valid_gt_mask, error_map, np.nan)
    valid_errors = error_map[valid_gt_mask]
    mean_error = valid_errors.mean() if valid_errors.size > 0 else 0
    max_error = valid_errors.max() if valid_errors.size > 0 else 0

    im5 = axes[2, 0].imshow(error_map, cmap="hot")
    axes[2, 0].set_title(f"Absolute Error (mean: {mean_error:.2f}, max: {max_error:.2f})")
    cbar5 = plt.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)
    cbar5.set_label('Error (pixels)', rotation=270, labelpad=15)

    # Error histogram
    axes[2, 1].hist(valid_errors.flatten(), bins=50, edgecolor='black')
    axes[2, 1].set_xlabel('Absolute Error (pixels)')
    axes[2, 1].set_ylabel('Pixel Count')
    axes[2, 1].set_title(f'Error Distribution (EPE: {error_map.mean():.3f})')
    axes[2, 1].grid(True, alpha=0.3)

    # Add percentage of pixels below certain thresholds
    total_pixels = valid_gt_mask.sum()
    pct_1px = (valid_errors < 1.0).sum() / total_pixels * 100 if total_pixels > 0 else 0
    pct_3px = (valid_errors < 3.0).sum() / total_pixels * 100 if total_pixels > 0 else 0
    axes[2, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label=f'<1px: {pct_1px:.1f}%')
    axes[2, 1].axvline(x=3.0, color='g', linestyle='--', alpha=0.5, label=f'<3px: {pct_3px:.1f}%')
    axes[2, 1].legend()

    for i, ax in enumerate(axes.flat):
        if i != 5:  # Keep axis on for histogram
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/{folder_name}/comparison_{val_id}.png")
    plt.close()



# --- Dataset Configuration ---
# Centralized configuration for each dataset to keep the main function generic.
DATASET_CONFIGS = {
    'eth3d': {
        'class': ETH3DStereo,
        'args': {},
        'epe_threshold': 1.0,
    },
    'kitti': {
        'class': KITTI15,
        'args': {'mode': 'training'},
        'epe_threshold': 3.0,
    },
    'middlebury': {
        'class': MiddleburyEval3,
        'args': {'resolution': 'Q'},
        'epe_threshold': 2.0,
    },
    'things': {
        'class': FlyingThings3D,
        'args': {'split': 'val'},
        'epe_threshold': 1.0,
        # Special filter for FlyingThings3D as per the original code
        'filter_fn': lambda disp: disp.abs() < 192
    },
}

@torch.no_grad()
def validate_dataset(
    model: torch.nn.Module,
    dataset_name: str,
    device: str, 
    mixed_prec: bool = True, 
    save_outputs: bool = False,
    **model_kwargs: Any,
    ):
    """
    Performs validation on a specified dataset using a generic, config-driven approach.

    Args:
        model: The PyTorch model to evaluate.
        dataset_name: The name of the dataset (e.g., 'kitti', 'eth3d').
        device: The device to run evaluation on ('cuda' or 'cpu').
        mixed_prec: Whether to use automatic mixed precision.
        save_outputs: Whether to save visualization images.
        **model_kwargs: Additional keyword arguments for the model's forward pass (e.g., iters_s8).

    Returns:
        A dictionary containing the calculated metrics (EPE and D1).
    """
    # Put the model in evaluation mode
    model.eval()
    
    # Build validation dataset
    config = DATASET_CONFIGS[dataset_name]
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    dataset_args = config.get('args', {})
    dataset_args['transform'] = val_transform
    val_dataset = config['class'](**dataset_args)

    # Assign constants and log
    logging.info(f"Evaluating on {dataset_name.upper()}. Total images: {len(val_dataset)}")
    val_epe, val_d1, val_thres, valid_samples = 0, 0, 0, 0

    for val_id in tqdm(range(len(val_dataset)), desc=f"Validating on {dataset_name.upper()}"):
        data = val_dataset[val_id]
        image1_inp, image2_inp, disp_gt = data['left'], data['right'], data['disp']
        
        # Make the valid mask
        valid_gt = disp_gt > 0
        if not valid_gt.any():
            continue

        # Convert inputs to half precision and pad the images. 
        image1 = image1_inp.half()
        image2 = image2_inp.half()
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        # Intialize model parameters
        model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

        # Inference
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=mixed_prec):
            results = model(image1, image2, **model_kwargs)
       
        disp_pred = results[-1] # Shape [1, H, W]
        disp_pred = padder.unpad(disp_pred)[0].squeeze(0).cpu() # Shape [H, W]
        assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)

        # Save the outputs
        if save_outputs:
            save_outputs_func(image1_inp, image2_inp, disp_gt, disp_pred, val_id, dataset_name)

        # Compute Metrics
        epe = F.l1_loss(disp_gt[valid_gt], disp_pred[valid_gt], reduction='mean')
        d1 = d1_metric(disp_pred, disp_gt, valid_gt)
        thres = thres_metric(disp_pred, disp_gt, valid_gt, config['epe_threshold'])
        
        valid_samples += 1
        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres += thres.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples
    mean_thres = val_thres / valid_samples

    # print(f"Validation ETH3D: EPE: {mean_epe} || D1: {mean_d1} || {config['epe_threshold']}px Error: {mean_thres}")
    return {'epe': mean_epe, 'd1': mean_d1, 'thresh': mean_thres}

# --- Simplified Validation Functions (API) ---

def validate_eth3d(model, device, **kwargs):
    """Perform validation on the ETH3D dataset."""
    return validate_dataset(model, 'eth3d', device, iters_s16=1, iters_s8=8, **kwargs)

def validate_kitti(model, device, **kwargs):
    """Perform validation on the KITTI-2015 dataset."""
    return validate_dataset(model, 'kitti', device, **kwargs)

def validate_middlebury(model, device, **kwargs):
    """Perform validation on the Middlebury-V3 dataset."""
    return validate_dataset(model, 'middlebury', device, **kwargs)

def validate_things(model, device, **kwargs):
    """Perform validation on the FlyingThings3D dataset."""
    return validate_dataset(model, 'things', device, **kwargs)
