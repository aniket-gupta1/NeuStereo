import sys
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, config, stage, device, mixed_prec=True):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    val_dataset = build_dataset(config, stage, split="train")

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1.half()
        image2 = image2.half()

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        
        model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            results = model(image1, image2)
        
        flow_pr = results[-1]

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
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'epe': epe, 'd1': d1}


@torch.no_grad()
def validate_things(model, config, stage, device, mixed_prec=True):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = build_dataset(config, stage, split="val")

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        # plt.imsave('image1.png', image1.permute(1,2,0).numpy()/255.0)
        # plt.imsave('image2.png', image2.permute(1,2,0).numpy()/255.0)
        # plt.imsave('disp_gt.png', flow_gt.permute(1,2,0).numpy().squeeze(), cmap='jet')

        image1 = image1.half()
        image2 = image2.half()

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            results = model(image1, image2)
        
        flow_pr = results[-1]
        # pdb.set_trace()
        # plt.imsave('disp_pred.png', flow_pr[0].permute(1,2,0).cpu().numpy().squeeze(), cmap='jet')

        # raise ValueError


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
def validate_middlebury(model, config, stage, device, mixed_prec=True):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    val_dataset = build_dataset(config, stage, split="2014")

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1.half()
        image2 = image2.half()

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        
        model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            results = model(image1, image2)
        
        flow_pr = results[-1]
        
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