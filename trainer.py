import torch
from evaluate_stereo import *
from utils.utils import load_config
from utils.stereo_metric import epe_metric, d1_metric, thres_metric
import os
import pdb
import torch.nn.functional as F

class Trainer():
    def __init__(self, cfg, args, logger, device, tensorboard_writer) -> None:
        self.cfg = cfg
        self.args = args
        self.logger = logger
        self.device = device
        self.tensorboard_writer = tensorboard_writer

        # Make the checkpoint directory
        os.makedirs(self.cfg.logdir + "/checkpoints", exist_ok=True)

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

    def loss_func(self, pred_disps, gt_disp, mask, max_disp=400, gamma=0.9):
        # Compute Smooth L1 loss with GT disparity 
        loss_weights = [0.9 ** (len(pred_disps) - 1 - power) for power in
                            range(len(pred_disps))]
        disp_loss = 0
        
        for k in range(len(pred_disps)):
            pred_disp = pred_disps[k]
            weight = loss_weights[k]

            curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask], reduction='mean')
            disp_loss += weight * curr_loss

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
        # Step 1: Set the model to train mode
        model.train()

        actual_model = self.get_model(model)
        actual_model.init_bhwd(self.cfg.batch_size, 384, 768, self.device)

        # Step 2: Iterate over the training loader
        for i, sample in enumerate(train_loader):
            img1, img2, disp_gt = [sample[x].to(self.device) for x in sample]
            disp_gt = disp_gt.unsqueeze(1)
            valid = (disp_gt > 0) & (disp_gt < self.cfg.max_disp)
            if not valid.any():
                continue

            img1 = img1.half()
            img2 = img2.half()
            # actual_model = self.get_model(model)
            # actual_model.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], self.device)
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=True):
                disp_preds = model(img1, img2, iters_s16=1, iters_s8=8)
                loss, metrics = self.loss_func(disp_preds, disp_gt, valid, self.cfg.max_disp)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if self.args.local_rank == 0 and i%10==0:
                self.logger.info(f"Epoch: {epoch_num:3d}, Step: {i:6d}, EPE: {metrics['epe']:7.3f}, D1: {metrics['d1']:3.2f}, 1px: {metrics['1px_error']:3.2f}, 2px: {metrics['2px_error']:3.2f}, 3px: {metrics['3px_error']:3.2f}")
                
                # Add the loss, epe, mag and learning rate to tensorboard
                self.tensorboard_writer.add_scalar('Train/Loss', loss.item(), epoch_num * len(train_loader) + i)
                self.tensorboard_writer.add_scalar('Train/EPE', metrics['epe'], epoch_num * len(train_loader) + i)
                self.tensorboard_writer.add_scalar('Train/D1', metrics['d1'], epoch_num * len(train_loader) + i)
                self.tensorboard_writer.add_scalar('Train/1px_error', metrics['1px_error'], epoch_num * len(train_loader) + i)
                self.tensorboard_writer.add_scalar('Train/2px_error', metrics['2px_error'], epoch_num * len(train_loader) + i)
                self.tensorboard_writer.add_scalar('Train/3px_error', metrics['3px_error'], epoch_num * len(train_loader) + i)
                self.tensorboard_writer.add_scalar('Train/LR', optimizer.param_groups[-1]['lr'], epoch_num * len(train_loader) + i)

    def val(self, model, epoch_num=1):
        for stage in self.cfg.val_stage:
            if stage=="flyingthings":
                results = validate_things(model, device=self.device)
                if self.args.local_rank == 0:
                    self.logger.info(f"For {stage}: EPE: {results['epe']:7.3f} || d1: {results['d1']:3.2f} || 1px error: {results['thresh']:3.2f}")
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-EPE', results['epe'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-d1', results['d1'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-1px_error', results['thresh'], epoch_num)

            if stage=="kitti":
                results = validate_kitti(model, device=self.device, save_outputs=False)
                if self.args.local_rank == 0:    
                    self.logger.info(f"For {stage}: EPE: {results['epe']:7.3f} || d1: {results['d1']:3.2f} || 3px error: {results['thresh']:3.2f}")
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-EPE', results['epe'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-d1', results['d1'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-3px_error', results['thresh'], epoch_num)

            if stage=="eth3d":
                results = validate_eth3d(model, device=self.device, save_outputs=False)
                if self.args.local_rank == 0:    
                    self.logger.info(f"For {stage}: EPE: {results['epe']:7.3f} || d1: {results['d1']:3.2f} || 1px error: {results['thresh']:3.2f}")
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-EPE', results['epe'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-d1', results['d1'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-1px_error', results['thresh'], epoch_num)

            if stage=="middlebury":
                results = validate_middlebury(model, device=self.device, save_outputs=False)
                if self.args.local_rank == 0:    
                    self.logger.info(f"For {stage}: EPE: {results['epe']:7.3f} || d1: {results['d1']:3.2f} || 2px error: {results['thresh']:3.2f}")
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-EPE', results['epe'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-d1', results['d1'], epoch_num)
                    self.tensorboard_writer.add_scalar(f'Val/{stage}-2px_error', results['thresh'], epoch_num)

        # if self.cfg.plot_iterations_curve:
        #     plot_iterations_curve(actual_model, self.device, datasets=self.cfg.val_stage)

    def fit(self, model, train_loader, train_sampler):
        # Step 1: Configure the optimizer, mixed precision, learning rate scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        
        # Step 2: Run the training loop
        for epoch_num in range(1, self.cfg.num_epochs+1):
            self.train(model, train_loader, optimizer, scaler, epoch_num)

            # Save the checkpoint before validation
            self.save_checkpoint(model, optimizer, epoch_num)

            if epoch_num % self.cfg.val_freq == 0 and not self.args.distributed:
                self.val(model, epoch_num)