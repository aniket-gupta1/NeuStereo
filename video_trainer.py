import torch
from evaluate_stereo import *
from utils.utils import load_config
from utils.losses import EPE_loss, edge_aware_smoothness_loss, second_order_smoothness_loss, photometric_reconstruction_loss
from utils.stereo_metric import epe_metric, d1_metric, thres_metric
import os
import pdb
import torch.nn.functional as F
from NeuStereo_video.softmax_warp import GeometricWarp

# Defensive import for mixed precision
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    print("Warning: torch.cuda.amp not available. Running in float32. Check PyTorch installation for CUDA support.")
    AMP_AVAILABLE = False
    class autocast:
        def __init__(self, enabled=False): pass
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    class GradScaler:
        def __init__(self, enabled=False): self.enabled = enabled
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass

class Trainer():
    def __init__(self, cfg, args, logger, device, tensorboard_writer) -> None:
        self.cfg = cfg
        self.args = args
        self.logger = logger
        self.device = device
        self.tensorboard_writer = tensorboard_writer
        self.use_amp = args.enable_amp and AMP_AVAILABLE and self.device.type == 'cuda'
        self.debug = getattr(args, 'debug', False) # Safely get debug flag
        os.makedirs(self.cfg.logdir + "/checkpoints", exist_ok=True)

    def check_tensor(self, tensor, name, check_stats=True):
        """Enhanced tensor checker, only active if self.debug is True."""
        if not self.debug:
            return

        if tensor is None:
            print(f"DEBUG: 🟡 {name} is None")
            return
        if not isinstance(tensor, torch.Tensor):
            print(f"DEBUG: 🟡 {name} is not a tensor, but type {type(tensor)}")
            return
        
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan or has_inf:
            status = "🔴 NaN" if has_nan else "🔴 Inf"
            print(f"DEBUG: {status} found in '{name}' | Shape: {tensor.shape}")
            raise ValueError(f"Numerical instability detected in tensor: {name}")
        elif check_stats:
            print(f"DEBUG: ✅ '{name}' is valid | Shape: {tensor.shape} | Mean: {torch.mean(tensor.float()):.4f} | Max: {torch.max(tensor.float()):.4f} | Min: {torch.min(tensor.float()):.4f}")

    def get_model(self, model):
        """Get the actual model from DDP wrapper if needed"""
        return model.module if hasattr(model, 'module') else model

    def save_checkpoint(self, model, optimizer, epoch_num):
        actual_model = self.get_model(model)
        
        checkpoint = {
            'model': actual_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_num
        }

        torch.save(checkpoint, self.cfg.logdir + f"/checkpoints/epoch_{epoch_num}.pth")

    def loss_func(self, left, right, pred_disps, gt_disp, mask, max_disp=400, gamma=0.9):
        # Compute Smooth L1 loss with GT disparity 
        disp_loss = EPE_loss(pred_disps, gt_disp, mask, gamma)

        # Compute First Order Smoothness Loss (Edge Loss)
        # edge_loss = edge_aware_smoothness_loss(pred_disps, left)

        # Compute Second Order Smoothness Loss (Planar Loss)
        # planar_loss = second_order_smoothness_loss(pred_disps, left)
        # planar_loss = second_order_smoothness_loss(gt_disp, left)

        # Compute Photometric Loss (By warping left image with the predicted disparity and taking L1 loss)
        # photometric_loss = photometric_reconstruction_loss(pred_disps, left, right, ssim_weight=0.85)
        # photometric_loss = photometric_reconstruction_loss(gt_disp.half(), left, right, ssim_weight=0.85)

        # pdb.set_trace()

        total_loss = disp_loss

        # Compute Metrics
        pred_disp = pred_disps[-1]
        epe = epe_metric(pred_disp, gt_disp, mask)
        d1 = d1_metric(pred_disp, gt_disp, mask) * 100
        thresh_1px = thres_metric(pred_disp, gt_disp, mask, thres=1) * 100
        thresh_2px = thres_metric(pred_disp, gt_disp, mask, thres=2) * 100
        thresh_3px = thres_metric(pred_disp, gt_disp, mask, thres=3) * 100

        metrics = {
            'epe': epe, 
            'd1': d1,
            '1px_error': thresh_1px,
            '2px_error': thresh_2px,
            '3px_error': thresh_3px,
        }

        return total_loss, metrics 

    def train(self, model, train_loader, optimizer, scaler, epoch_num):
        model.train()
        warper = GeometricWarp().to(self.device)
        warper.eval()
        actual_model = self.get_model(model)
        actual_model.init_bhwd(self.cfg.batch_size, 384, 768, self.device, amp=self.use_amp)

        for step_num, sample in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            total_loss_sequence = None
            
            prev_pose, prev_contexts, prev_disp = None, None, None

            # `video_collate_fn` returns tensors with shape [B, T, ...].
            # Iterate over timesteps (T) and slice the batch at the time index.
            T = sample['left'].shape[1]
            for timestep in range(T):
                # Convert everything to cuda first by slicing the time dimension.
                # Ignore any extra keys (e.g. 'frame_mask') and handle optional fields.
                left = sample['left'][:, timestep].to(self.device)
                right = sample['right'][:, timestep].to(self.device)

                disp_gt = None
                if sample.get('disp') is not None:
                    disp_gt = sample['disp'][:, timestep].to(self.device)

                pose = None
                if sample.get('pose') is not None:
                    pose = sample['pose'][:, timestep].to(self.device)

                intrinsics = None
                if sample.get('intrinsics') is not None:
                    intrinsics = sample['intrinsics'][:, timestep].to(self.device)

                # Make valid pixels mask
                if disp_gt is None:
                    # No ground-truth disparity for this batch/timestep -> skip
                    continue
                if intrinsics is None:
                    # Intrinsics required for warping; skip if missing
                    continue

                disp_gt = disp_gt.unsqueeze(1)
                valid = (disp_gt > 0) & (disp_gt < self.cfg.max_disp)
                if not valid.any():
                    continue

                # Convert images to half precision
                left_input = left.half() if self.use_amp else left.float()
                right_input = right.half() if self.use_amp else right.float()

                with autocast(enabled=self.use_amp):
                    warped_disp, warped_contexts = None, None
                    if timestep > 0 and prev_pose is not None:
                        if self.debug: print(f"\n--- Timestep {timestep}: Entering Warping ---")
                        with torch.no_grad():
                            self.check_tensor(prev_disp, "Warp Input: prev_disp")
                            inv_prev_pose = torch.linalg.inv(prev_pose)
                            self.check_tensor(inv_prev_pose, "Warp Intermediate: inv_prev_pose")
                            relative_pose = pose @ inv_prev_pose
                            self.check_tensor(relative_pose, "Warp Intermediate: relative_pose")
                            
                            B = pose.shape[0]
                            fx = intrinsics[:, 0]
                            cx = intrinsics[:, 2]
                            cy = intrinsics[:, 3]

                            Q = torch.zeros(B, 4, 4, device=left.device, dtype=left_input.dtype)
                            Q[:, 0, 0] = 1.0
                            Q[:, 0, 3] = -cx
                            Q[:, 1, 1] = 1.0
                            Q[:, 1, 3] = -cy
                            Q[:, 2, 3] = fx
                            Q[:, 3, 2] = 1.0 / 65.0  # baseline = 65mm
                            T_geo = Q @ relative_pose @ torch.linalg.inv(Q)
                            self.check_tensor(T_geo, "Warp Intermediate: T_geo")

                            warped_contexts, warped_disp = warper(prev_contexts, prev_disp, T_geo)
                            self.check_tensor(warped_disp, "Warp Output: warped_disp")
                            if self.debug: print("--- Exiting Warping ---")

                    disp_preds, contexts = model(left_input, right_input, warped_disp, warped_contexts, iters_s16=1, iters_s8=8)
                    
                    prev_pose, prev_contexts, prev_disp = pose, contexts, disp_preds[-1]

                    loss, metrics = self.loss_func(left, right, disp_preds, disp_gt, valid, self.cfg.max_disp)
                    # Accumulate losses as a tensor. Initialize on first real loss.
                    if total_loss_sequence is None:
                        total_loss_sequence = loss
                    else:
                        total_loss_sequence = total_loss_sequence + loss

            if total_loss_sequence is not None:
                if self.use_amp:
                    scaler.scale(total_loss_sequence).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss_sequence.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            if self.args.local_rank == 0 and step_num % 10 == 0 and total_loss_sequence is not None:
                # Log the full set of original metrics (only if we computed a loss)
                try:
                    seq_loss_val = total_loss_sequence.item()
                except Exception:
                    seq_loss_val = float(total_loss_sequence)

                log_msg = (f"Epoch: {epoch_num:3d}, Step: {step_num:6d}, Seq Loss: {seq_loss_val:.4f}, "
                           f"EPE: {metrics['epe']:.3f}, D1: {metrics['d1']:.2f}, "
                           f"1px: {metrics['1px_error']:.2f}, 3px: {metrics['3px_error']:.2f}")
                self.logger.info(log_msg)
                self.tensorboard_writer.add_scalar('Train/Loss', seq_loss_val, epoch_num * len(train_loader) + step_num)
                self.tensorboard_writer.add_scalar('Train/EPE', metrics['epe'], epoch_num * len(train_loader) + step_num)
    
                # Add the loss, epe, mag and learning rate to tensorboard
                self.tensorboard_writer.add_scalar('Train/Loss', loss.item(), epoch_num * len(train_loader) + step_num)
                self.tensorboard_writer.add_scalar('Train/EPE', metrics['epe'], epoch_num * len(train_loader) + step_num)
                self.tensorboard_writer.add_scalar('Train/D1', metrics['d1'], epoch_num * len(train_loader) + step_num)
                self.tensorboard_writer.add_scalar('Train/1px_error', metrics['1px_error'], epoch_num * len(train_loader) + step_num)
                self.tensorboard_writer.add_scalar('Train/2px_error', metrics['2px_error'], epoch_num * len(train_loader) + step_num)
                self.tensorboard_writer.add_scalar('Train/3px_error', metrics['3px_error'], epoch_num * len(train_loader) + step_num)
                self.tensorboard_writer.add_scalar('Train/LR', optimizer.param_groups[-1]['lr'], epoch_num * len(train_loader) + step_num)

    def val(self, model, epoch_num=1):
        iters_s16, iters_s8 = 40,40
        self.logger.info(f"Using s16 refinements: {iters_s16} and s8 refinements: {iters_s8}")
    
        for stage in self.cfg.val_stage:
            if stage=="flyingthings":
                results = validate_things(model, device=self.device)
                if self.args.local_rank == 0:
                    self.logger.info(f"For {stage}: EPE: {results['epe']:7.3f} || d1: {results['d1']:3.2f} || 1px error: {results['thresh']:3.2f}")
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-EPE', results['epe'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-d1', results['d1'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-1px_error', results['thresh'], epoch_num)

            if stage=="kitti":
                results = validate_kitti(model, device=self.device, save_outputs=False, iters_s16=iters_s16, iters_s8=iters_s8)
                if self.args.local_rank == 0:    
                    self.logger.info(f"For {stage}: EPE: {results['epe']:7.3f} || d1: {results['d1']:3.2f} || 3px error: {results['thresh']:3.2f}")
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-EPE', results['epe'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-d1', results['d1'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-3px_error', results['thresh'], epoch_num)

            if stage=="eth3d":
                results = validate_eth3d(model, device=self.device, save_outputs=False, iters_s16=iters_s16, iters_s8=iters_s8)
                if self.args.local_rank == 0:    
                    self.logger.info(f"For {stage}: EPE: {results['epe']:7.3f} || d1: {results['d1']:3.2f} || 1px error: {results['thresh']:3.2f}")
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-EPE', results['epe'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-d1', results['d1'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-1px_error', results['thresh'], epoch_num)

            if stage=="middlebury":
                results = validate_middlebury(model, device=self.device, save_outputs=False, iters_s16=iters_s16, iters_s8=iters_s8)
                if self.args.local_rank == 0:    
                    self.logger.info(f"For {stage}: EPE: {results['epe']:7.3f} || d1: {results['d1']:3.2f} || 2px error: {results['thresh']:3.2f}")
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-EPE', results['epe'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-d1', results['d1'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-2px_error', results['thresh'], epoch_num)

            if stage=="inference":
                inference(model, device=self.device, save_outputs=False, iters_s16=1, iters_s8=8)

    def fit(self, model, train_loader, train_sampler):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scaler = GradScaler(enabled=self.use_amp)
        
        for epoch_num in range(1, self.cfg.num_epochs + 1):
            if train_sampler is not None and self.args.distributed:
                train_sampler.set_epoch(epoch_num)
            self.train(model, train_loader, optimizer, scaler, epoch_num)
            if self.args.local_rank == 0:
                self.save_checkpoint(model, optimizer, epoch_num)

            if epoch_num % self.cfg.val_freq == 0 and not self.args.distributed:
                self.val(model, epoch_num)