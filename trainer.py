import torch

class Trainer():
    def __init__(self, cfg, args, logger, device, tensorboard_writer) -> None:
        self.cfg = cfg
        self.args = args
        self.logger = logger
        self.device = device
        self.tensorboard_writer = tensorboard_writer

    def save_checkpoint(self, model, optimizer, epoch_num):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_num
        }

        torch.save(checkpoint, self.cfg.checkpoint_dir + f"epoch_{epoch_num}.pth")

    def loss_func(self, disp_preds, disp_gt, valid, max_disp=400, gamma=0.9):
        n_predictions = len(disp_preds)
        disp_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()  # [B, H, W]
        valid = (valid >= 0.5) & (mag < max_disp)

        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()
            disp_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()

        if valid.max() < 0.5:
            pass

        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            'mag': mag.mean().item()
        }

        return disp_loss, metrics

    def train(self, model, train_loader, optimizer, scaler, epoch_num):
        # Step 1: Set the model to train mode
        model.train()

        # Step 2: Iterate over the training loader
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()

            img1, img2, disp_gt, valid = [x.to(self.device) for x in sample]

            img1 = img1.half()
            img2 = img2.half()

            model.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], self.device)

            with torch.cuda.amp.autocast(enabled=True):
                disp_preds = model(img1, img2, iters_s16=4, iters_s8=7)
                loss, metrics = self.loss_func(disp_preds, disp_gt, valid, self.cfg.max_disp)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            bad_grad = False
            for name, param in model.named_parameters():
                if not torch.all(torch.isfinite(param.grad)):
                    bad_grad = True
                if bad_grad:
                    print(name, param.grad.mean().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            self.logger.info(f"Epoch: {epoch_num}, Step: {i}, EPE: {round(metrics['epe'], 3)}, Mag: {round(metrics['mag'], 3)}, LR: {optimizer.param_groups[-1]['lr']}")
            
            # Add the loss, epe, mag and learning rate to tensorboard
            self.tensorboard_writer.add_scalar('Train/Loss', loss.item(), epoch_num * len(train_loader) + i)
            self.tensorboard_writer.add_scalar('Train/EPE', metrics['epe'], epoch_num * len(train_loader) + i)
            self.tensorboard_writer.add_scalar('Train/Mag', metrics['mag'], epoch_num * len(train_loader) + i)
            self.tensorboard_writer.add_scalar('Train/LR', optimizer.param_groups[-1]['lr'], epoch_num * len(train_loader) + i)


    def val(self, model, val_loader, epoch_num):
        # Step 1: Set the model to eval
        model.eval()

        #TODO: Get separate evals for different datasets.


        # Step 2: Iterate over the validation loader
        for i, sample in enumerate(val_loader):
            img1, img2, disp_gt, valid = [x.to(self.device) for x in sample]

            img1 = img1.half()
            img2 = img2.half()

            model.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], self.device)

            with torch.cuda.amp.autocast(enabled=True):
                disp_preds = model(img1, img2)
                loss, metrics = self.loss_func(disp_preds, disp_gt, valid, model.cfg.max_disp)
        
            self.logger.info(f"Epoch: {epoch_num}, Step: {i}, EPE: {round(metrics['epe'], 3)}, Mag: {round(metrics['mag'], 3)}")

            # Add the loss, epe and mag to tensorboard
            self.tensorboard_writer.add_scalar('Val/Loss', loss.item(), epoch_num * len(val_loader) + i)
            self.tensorboard_writer.add_scalar('Val/EPE', metrics['epe'], epoch_num * len(val_loader) + i)
            self.tensorboard_writer.add_scalar('Val/Mag', metrics['mag'], epoch_num * len(val_loader) + i)

    def fit(self, model, train_loader, val_loader, train_sampler):
        # Step 1: Configure the optimizer, mixed precision, learning rate scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        
        # Step 2: Run the training loop
        for epoch_num in range(1, self.cfg.num_epochs+1):
            self.train(model, train_loader, optimizer, scaler, epoch_num)

            if epoch_num % self.cfg.val_freq == 0:
                self.val(model, val_loader, epoch_num)
        
        # Step 3: Save the model
        self.save_checkpoint(model, optimizer, epoch_num)






