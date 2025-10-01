import torch
from evaluate_stereo import *
from utils.utils import load_config
from utils.losses import EPE_loss, edge_aware_smoothness_loss, second_order_smoothness_loss, photometric_reconstruction_loss
from utils.stereo_metric import epe_metric, d1_metric, thres_metric
import os
import pdb
import torch.nn.functional as F
from NeuStereo_video.softmax_warp import GeometricWarp
from sanity_check import plot_video_stereo_debug

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
        # Step 1: Set the model to train mode
        model.train()
        warper = GeometricWarp()
        warper.eval()

        actual_model = self.get_model(model)
        actual_model.init_bhwd(self.cfg.batch_size, 384, 768, self.device)

        # Step 2: Iterate over the training loader
        for step_num, sample in enumerate(train_loader):
            # left_images, right_images, gt_disparities, predicted_disparities, warped_disp_t1, warped_features_t1 = [], [], [], [], None, None
            
            # Loop over all the video frames
            for timestep in range(len(sample['left'])):
                # Convert everything to cuda first
                left, right, disp_gt, pose, intrinsics = [sample[x][timestep].to(self.device) for x in sample]
                # left = sample['left'][timestep].to(self.device)
                # right = sample['right'][timestep].to(self.device)
                # disp_gt = sample['disp'][timestep].to(self.device)
                # pose = sample['pose'][timestep].to(self.device)
                # intrinsics = sample['intrinsics'][timestep].to(self.device)

                # Make valid pixels mask
                disp_gt = disp_gt.unsqueeze(1)
                valid = (disp_gt > 0) & (disp_gt < self.cfg.max_disp)
                if not valid.any():
                    continue

                # Convert images to half precision
                left = left.half()
                right = right.half()

                # left_images.append(left.clone())
                # right_images.append(right.clone())
                # gt_disparities.append(disp_gt.clone())

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=True):
                    if timestep==0:
                        # Run forward pass
                        disp_preds, contexts = model(left, right, warped_prev_disp=None, warped_prev_contexts=None, iters_s16=1, iters_s8=8)
                        prev_pose = pose
                        disp_gt_prev = disp_gt

                        # predicted_disparities.append(disp_preds[-1])
                        
                    else:
                        with torch.no_grad():
                            # Compute relative pose
                            relative_pose = pose @ torch.linalg.inv(prev_pose)

                            # Compute the disparity transformation matrix
                            batch_size = pose.shape[0]
                            # Extract intrinsics for each batch element
                            fx = intrinsics[:, 0]  # [B]
                            cx = intrinsics[:, 2]  # [B]
                            cy = intrinsics[:, 3]  # [B]

                            # Create batched Q matrix [B, 4, 4]
                            Q = torch.zeros(batch_size, 4, 4, device=left.device)
                            Q[:, 0, 0] = 1.0
                            Q[:, 0, 3] = -cx
                            Q[:, 1, 1] = 1.0
                            Q[:, 1, 3] = -cy
                            Q[:, 2, 3] = fx
                            Q[:, 3, 2] = 1.0 / 65.0  # baseline = 65mm
                            T_geo = Q @ relative_pose @ torch.linalg.inv(Q)

                            # Warp disparity and context
                            warped_contexts, warped_disp = warper(contexts, disp_preds[-1], T_geo)
                            # warped_contexts, warped_disp = warper(contexts, disp_gt_prev.half(), T_geo)

                        # Run forward pass
                        disp_preds, contexts = model(left, right, warped_prev_disp=warped_disp, warped_prev_contexts=warped_contexts, iters_s16=1, iters_s8=8)
                        prev_pose = pose

                        # predicted_disparities.append(disp_preds[-1])
                        # warped_disp_t1 = warped_disp
                        # warped_features_t1 = warped_contexts

                        # if timestep==1:
                        #     plot_video_stereo_debug(
                        #         left_img_t0=left_images[0],
                        #         right_img_t0=right_images[0],
                        #         gt_disp_t0=gt_disparities[0],
                        #         pred_disp_t0=predicted_disparities[0],
                                
                        #         left_img_t1=left_images[1],
                        #         right_img_t1=right_images[1],
                        #         gt_disp_t1=gt_disparities[1],
                        #         pred_disp_t1=predicted_disparities[1],
                                
                        #         warped_disp_t1=warped_disp_t1,
                        #         warped_features_t1=warped_features_t1,
                                
                        #         save_path=f"debug/step_{step_num}_warp_check.png"
                        #     )

                        # raise ValueError
                    
                    # Compute Loss
                    loss, metrics = self.loss_func(left, right, disp_preds, disp_gt, valid, self.cfg.max_disp)
                
                # Run backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            if self.args.local_rank == 0 and step_num%10==0:
                self.logger.info(f"Epoch: {epoch_num:3d}, Step: {step_num:6d}, EPE: {metrics['epe']:7.3f}, D1: {metrics['d1']:3.2f}, 1px: {metrics['1px_error']:3.2f}, 2px: {metrics['2px_error']:3.2f}, 3px: {metrics['3px_error']:3.2f}")
                
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