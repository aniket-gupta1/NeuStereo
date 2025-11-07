import torch
import torch.distributed
from torch.utils.data import DataLoader
from easydict import EasyDict
import argparse
import os
from dataloader.video_datasets import build_dataset
from NeuStereo_video.neustereo_video import NeuStereo
from video_trainer import Trainer
from utils.utils import prepare_logger, load_config
from torch.utils.tensorboard import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser()

    # Tensorboard summary name
    parser.add_argument('--exp_name', default="test_single_mask", type=str)
    parser.add_argument('--logdir', default="logs/", type=str)
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--TIMEIT', default=False, type=bool)
    parser.add_argument('--dev', action='store_true', default=False)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--enable_amp', action='store_true', default=False)
    parser.add_argument('--val_only', default=False, action='store_true')

    # Model

    # Dataset
    parser.add_argument('--checkpoint_dir', default="checkpoints/", type=str)
    parser.add_argument('--stage', default=['kitti'], type=str, nargs='+')
    parser.add_argument('--val_dataset', default=[''], type=str, nargs='+')

    # Training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=0.005, type=float)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--epochs', default=500, type=int)

    # Resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str)

    # Distributed training
    parser.add_argument('--distributed', action='store_true', default=False)

    # Output
    parser.add_argument('--save_output', default=False, type=bool)

    return parser

def setup_dataloaders(cfg, args, logger):
    args.stage = cfg.stage
    train_dataset = build_dataset(args)

    if args.local_rank == 0:
        if isinstance(cfg.stage, list):
            for stage in cfg.stage:
                logger.info(f"Number of training samples in {stage}: {len(train_dataset.datasets[cfg.stage.index(stage)])}")
        else:
            logger.info(f"Total training samples: {len(train_dataset)}")

    if args.distributed:
        if torch.distributed.is_available():
            initialized = torch.distributed.is_initialized()
        else:
            initialized = False

        if initialized:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=args.local_rank)

    else:
        train_sampler = None

    shuffle = False #if args.distributed else True
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    return train_loader, train_sampler

def load_checkpoint_flexible(model, checkpoint_path, device, force_single_gpu=False):
    """
    Flexible checkpoint loader that adapts to current GPU setup
    
    Args:
        model: Your model
        checkpoint_path: Path to checkpoint
        device: Current device
        force_single_gpu: Force single GPU mode even if multiple GPUs available
    """
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Detect if checkpoint is from multi-GPU training
    checkpoint_is_multigpu = any(k.startswith('module.') for k in state_dict.keys())
    
    # Detect current setup
    current_is_multigpu = (torch.cuda.device_count() > 1 and 
                          hasattr(model, 'module')) or force_single_gpu == False
    
    print(f"Checkpoint is multi-GPU: {checkpoint_is_multigpu}")
    print(f"Current setup is multi-GPU: {current_is_multigpu}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Handle different scenarios
    if checkpoint_is_multigpu and not current_is_multigpu:
        # Multi-GPU checkpoint → Single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        
    elif not checkpoint_is_multigpu and current_is_multigpu:
        # Single GPU checkpoint → Multi-GPU
        print("Converting single GPU checkpoint to multi-GPU")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = f'module.{key}' if not key.startswith('module.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        
    else:
        # Same format, no conversion needed
        print("Checkpoint format matches current setup")
        pass
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        print("Partial loading completed")
    
    return model

def setup_model(cfg, args, logger, device):
    
    model = NeuStereo(cfg).to(device)
    # model = NeuFlow(cfg).to(device)
    if args.resume:
        model = load_checkpoint_flexible(model, args.resume, device, logger)
        logger.info(f"Loaded model from checkpoint: {args.resume}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    if args.local_rank == 0:
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    return model

def main(cfg, args, logger):
    # Params for distributed training
    if args.distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        if args.local_rank==0:
            logger.info(f"Using multi-gpu training. Total GPUS: {world_size}")

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make the tensorboard writer
    writer = SummaryWriter(log_dir=cfg.logdir+"/tensorboard/")

    # Setup the dataloaders
    if not args.val_only:
        train_loader, train_sampler = setup_dataloaders(cfg, args, logger)

    # Setup the model
    model = setup_model(cfg, args, logger, device)

    # Setup the trainer
    trainer = Trainer(cfg, args, logger, device, writer)

    # Train the model
    if args.val_only:
        if args.distributed:
            logger.info("Evaluation on multi-gpu not supported!!")
            raise ValueError
        trainer.val(model)
    else:
        trainer.fit(model, train_loader, train_sampler)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if args.config is None:
        if args.resume is None or not os.path.exists(args.resume):
            raise ValueError("--config needs to be supplied if not resuming from a checkpoint")
        else:
            resume_folder = args.resume if os.path.isdir(args.resume) else os.path.dirname(args.resume)
            args.config = os.path.normpath(os.path.join(resume_folder, "../config.yaml"))

            if os.path.exists(args.config):
                print(f"Using config file from checkpoint directory: {args.config}")
            elif not os.path.exists(args.config):
                raise ValueError(f"Config file not found in checkpoint directory: {args.config}")
    
    cfg = EasyDict(load_config(args.config))

    # Store different datasets to its own subdirectory
    # In case of training on multiple datasets, join their names with '_' and make a new directory
    # cfg.dataset is a list of dataset names
    # logdir_name = ""
    # if len(cfg.stage) == 1:
    #     logdir_name = cfg.stage[0]
    # else:
    #     for dataset_name in cfg.stage:
    #         logdir_name += dataset_name + "_"
    logdir_name = cfg.stage
    
    if args.dev:
        cfg.logdir = "logdev/"

    cfg.logdir = os.path.join(cfg.logdir, logdir_name) 
    args.logdir = cfg.logdir

    if args.exp_name is None and len(cfg.get("exp_name", "")) > 0:
        args.exp_name = cfg.exp_name
    
    logger, cfg.logdir = prepare_logger(args)
    if args.local_rank==0:
        logger.info('Output and logs will be saved to {}'.format(cfg.logdir))

    # Save the config file to the log directory
    config_out_fname = os.path.join(cfg.logdir, "config.yaml")
    with open(args.config, "r") as in_fid, open(config_out_fname, "w") as out_fid:
        out_fid.write(f"Original config file name: {args.config}\n")
        out_fid.write(in_fid.read())

    # Also save the current codebase to the log directory

    main(cfg, args, logger)