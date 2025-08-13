import sys
import os
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from datasets import build_dataset
from utils import InputPadder
import pdb
import matplotlib.pyplot as plt
import torch.utils.benchmark as benchmark

def count_parameters(model):
    actual_model = model.module if hasattr(model, 'module') else model
    
    return sum(p.numel() for p in actual_model.parameters() if p.requires_grad)

def benchmark_model_fixed(model, device):
    elapsed_list = []
    print("Working on image size: (512, 384)")
    for i in range(50):
        image1, image2 = torch.rand(3, 384, 512), torch.rand(3, 384, 512)
        image1 = image1[None].to(device)
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

def benchmark_model(model, val_dataset, device):
    elapsed_list = []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
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
        time_taken = timer.timeit(1).mean * 1000  # Convert to milliseconds
        elapsed_list.append(time_taken)

    return np.mean(elapsed_list)

def save_outputs_func(image1, image2, disp_gt, disp_pred, val_id, folder_name):
    # Make the output folder
    os.makedirs(f"outputs/{folder_name}", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Create a 2x2 grid
    # Convert tensors to NumPy and normalize if needed
    img1_np = image1.permute(1, 2, 0).numpy() / 255.0
    img2_np = image2.permute(1, 2, 0).numpy() / 255.0
    disp_gt_np = disp_gt.permute(1, 2, 0).numpy().squeeze()

    # Display images
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title("Image 1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title("Image 2")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(disp_gt_np, cmap="jet")
    axes[1, 0].set_title("Ground Truth Disp.")
    axes[1, 0].axis("off")

    pred_disp = disp_pred[0].permute(1,2,0).cpu().numpy().squeeze()
    axes[1, 1].imshow(pred_disp, cmap="jet")
    axes[1, 1].set_title("Predicted Disp.")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/{folder_name}/comparison_{val_id}.png")

@torch.no_grad()
def validate_eth3d(model, config, stage, device, mixed_prec=True, save_outputs=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    val_dataset = build_dataset(config, stage, split="train")

    logging.info(f"Evaluating on ETH3D. Total images: {len(val_dataset)}")
    out_list, epe_list = [], []

    # Benchmark the model for speed
    # elapsed_time = benchmark_model(model, val_dataset, device)
    # print(f"Elapsed time: {elapsed_time} ms")

    for val_id in range(len(val_dataset)):
        image1_inp, image2_inp, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1_inp.half()
        image2 = image2_inp.half()

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

        # Inference
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=mixed_prec):
            results = model(image1, image2)
       
        flow_pr = results[-1]
        # Save the outputs
        if save_outputs:
            save_outputs_func(image1_inp, image2_inp, flow_gt, flow_pr, val_id, 'eth3d')

        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        # logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'epe': epe, 'd1': d1}


@torch.no_grad()
def validate_kitti(model, config, stage, device, mixed_prec=True, save_outputs=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = build_dataset(config, stage, split="train")
    torch.backends.cudnn.benchmark = True
    logging.info(f"Evaluating on KITTI. Total images: {len(val_dataset)}")
    
    # Store individual metrics instead of arrays
    epe_list = []
    d1_list = []  # Store D1 percentages instead of boolean arrays
    
    for val_id in range(len(val_dataset)):
        image1_inp, image2_inp, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1_inp.half()
        image2 = image2_inp.half()
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)
        
        with torch.cuda.amp.autocast(enabled=mixed_prec):
            results = model(image1, image2)
            flow_pr = results[-1]
            
        if save_outputs:
            save_outputs_func(image1_inp, image2_inp, flow_gt, flow_pr, val_id, 'KITTI')
            
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 3.0)
        
        # Calculate metrics per image
        image_d1 = out[val].float().mean().item()  # D1 percentage for this image
        image_epe = epe_flattened[val].mean().item()  # EPE for this image
        
        # Store individual metrics
        epe_list.append(image_epe)
        d1_list.append(image_d1)
    
    # Calculate overall metrics
    epe = np.mean(epe_list)
    d1 = 100 * np.mean(d1_list)  # Convert to percentage
    
    print(f"Validation KITTI: EPE {epe}, D1 {d1}")
    return {'epe': epe, 'd1': d1}


@torch.no_grad()
def validate_things(model, config, stage, device, mixed_prec=True, save_outputs=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = build_dataset(config, stage, split="val")

    logging.info(f"Evaluating on FlyingThings3D. Total images: {len(val_dataset)}")

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1_inp, image2_inp, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1_inp.half()
        image2 = image2_inp.half()

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            results = model(image1, image2)
        
        flow_pr = results[-1]

        # Save the outputs
        if save_outputs:
            save_outputs_func(image1_inp, image2_inp, flow_gt, flow_pr, val_id, 'flyingthings')

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)

        if np.isnan(epe[val].mean().item()): #TODO: Fix this part
            continue
        
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'epe': epe, 'd1': d1}


@torch.no_grad()
def validate_middlebury(model, config, stage, device, mixed_prec=True, save_outputs=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    val_dataset = build_dataset(config, stage, split="2014")

    logging.info(f"Evaluating on Middlebury. Total images: {len(val_dataset)}")
    
    # # Benchmark the model for speed
    # elapsed_time = benchmark_model_fixed(model, device)
    # print(f"Elapsed time: {elapsed_time} ms")
    
    # elapsed_time = benchmark_model(model, val_dataset, device)
    # print(f"Elapsed time: {elapsed_time} ms")

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1_inp, image2_inp, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1_inp.half()
        image2 = image2_inp.half()

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            results = model(image1, image2)
        
        flow_pr = results[-1]

        # Save the outputs
        if save_outputs:
            save_outputs_func(image1_inp, image2_inp, flow_gt, flow_pr, val_id, 'middlebury')
        
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        # logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury: EPE {epe}, D1 {d1}")
    return {f'epe': epe, f'd1': d1}