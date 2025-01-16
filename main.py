import torch
import torch.distributed
from torch.utils.data import DataLoader
from easydict import EasyDict
import argparse
import os
from datasets import build_dataset, MultiDataset
from NeuStereo.neustereo import NeuStereo
from trainer import Trainer
from utils import prepare_logger, load_config

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
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
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
    train_dataset_list = []
    val_dataset_list = []
    config_dict = {}
   
    for stage in cfg.stage:
        config = load_config(f"configs/dataset/{stage}.yaml")
        train_dataset_i = build_dataset(config, stage, split="train")
        val_dataset_i = build_dataset(config, stage, split="val")

        train_dataset_list.append(train_dataset_i)
        val_dataset_list.append(val_dataset_i)

    train_dataset = MultiDataset(train_dataset_list)
    val_dataset = MultiDataset(val_dataset_list)

    if args.local_rank == 0:
        for i, stage in enumerate(args.stage):
            logger.info(f'Number of training samples in {stage}: {len(train_dataset_list[i])}')
            logger.info(f'Number of validation samples in {stage}: {len(val_dataset_list[i])}')

        logger.info(f"Total training samples: {len(train_dataset)}")
        logger.info(f"Total validation samples: {len(val_dataset)}")
    
    # If using distributed, we need to intialize distributed sampler
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

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=args.local_rank)
    else:
        train_sampler = None
        val_sampler = None

    # Initialize the dataloader
    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    # Load validation dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler
    )

    return train_loader, val_loader, train_sampler

def setup_model(cfg, args, logger):
    # Params for distributed training
    if args.distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.resume:
        model = NeuStereo(cfg.model).to(device)
        model.load_state_dict(torch.load(args.resume))
        logger.info(f"Loaded model from checkpoint: {args.resume}")
    
    # Setup the model
    model = NeuStereo(cfg.model).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    return model

def main(cfg, args, logger):
    # Setup the dataloaders
    train_loader, val_loader, train_sampler = setup_dataloaders(cfg, args, logger)

    # Setup the model
    model = setup_model(cfg, args, logger)

    # Setup the trainer
    trainer = Trainer(cfg, args, logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader, train_sampler)

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
            else:
                raise ValueError(f"Config file not found in checkpoint directory: {args.config}")
    
    cfg = EasyDict(load_config(args.config))

    # Store different datasets to its own subdirectory
    # In case of training on multiple datasets, join their names with '_' and make a new directory
    # cfg.dataset is a list of dataset names
    logdir_name = ""
    if len(cfg.stage) == 1:
        logdir_name = cfg.stage[0]
    else:
        for dataset_name in cfg.stage:
            logdir_name += dataset_name + "_"

    args.logdir = os.path.join(args.logdir, logdir_name) 

    if args.exp_name is None and len(cfg.get("exp_name", "")) > 0:
        args.exp_name = cfg.exp_name
    
    logger, args.log_path = prepare_logger(args)

    # Save the config file to the log directory
    config_out_fname = os.path.join(args.log_path, "config.yaml")
    with open(args.config, "r") as in_fid, open(config_out_fname, "w") as out_fid:
        out_fid.write(f"Original config file name: {args.config}\n")
        out_fid.write(in_fid.read())

    # Also save the current codebase to the log directory

    main(cfg, args, logger)