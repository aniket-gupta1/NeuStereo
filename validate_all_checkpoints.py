import torch
import os
import glob
import argparse
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import json
from easydict import EasyDict


# Import your evaluation functions
from evaluate_stereo import *
from utils.utils import load_config

def setup_logger(log_path):
    """Setup logger for the validation script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_checkpoint(checkpoint_path, model, device):
    """Load a checkpoint and return the model"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        
        # Extract epoch number if available
        epoch_num = None
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            epoch_num = checkpoint['epoch']
        else:
            # Try to extract from filename
            try:
                epoch_num = int(Path(checkpoint_path).stem.split('_')[-1])
            except:
                epoch_num = None
                
        return model, epoch_num
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None

def validate_checkpoint(model, datasets, device):
    """Validate a model on specified datasets"""
    results = {}
    model.eval()
    
    with torch.no_grad():
        for dataset in datasets:
            if dataset == "flyingthings":
                try:
                    dataset_results = validate_things(model, device=device)
                    results[dataset] = {
                        'epe': dataset_results['epe'],
                        'd1': dataset_results['d1'],
                        '1px_error': dataset_results['thresh']
                    }
                except Exception as e:
                    print(f"Error validating on {dataset}: {e}")
                    results[dataset] = None
                    
            elif dataset == "kitti":
                try:
                    dataset_results = validate_kitti(model, device=device, save_outputs=False)
                    results[dataset] = {
                        'epe': dataset_results['epe'],
                        'd1': dataset_results['d1'],
                        '3px_error': dataset_results['thresh']
                    }
                except Exception as e:
                    print(f"Error validating on {dataset}: {e}")
                    results[dataset] = None
                    
            elif dataset == "eth3d":
                try:
                    dataset_results = validate_eth3d(model, device=device, save_outputs=False)
                    results[dataset] = {
                        'epe': dataset_results['epe'],
                        'd1': dataset_results['d1'],
                        '1px_error': dataset_results['thresh']
                    }
                except Exception as e:
                    print(f"Error validating on {dataset}: {e}")
                    results[dataset] = None
                    
            elif dataset == "middlebury":
                try:
                    dataset_results = validate_middlebury(model, device=device, save_outputs=False)
                    results[dataset] = {
                        'epe': dataset_results['epe'],
                        'd1': dataset_results['d1'],
                        '2px_error': dataset_results['thresh']
                    }
                except Exception as e:
                    print(f"Error validating on {dataset}: {e}")
                    results[dataset] = None
                    
    return results

def main():
    parser = argparse.ArgumentParser(description='Validate all checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoints')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['eth3d', 'middlebury', 'kitti'],
                        choices=['flyingthings', 'kitti', 'eth3d', 'middlebury'],
                        help='Datasets to validate on')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                        help='Directory to save results')
    parser.add_argument('--checkpoint_pattern', type=str, default='*.pth',
                        help='Pattern to match checkpoint files')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use')
    parser.add_argument('--model_name', type=str, default='NeuStereo',
                        help='Model architecture name')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(os.path.join(args.output_dir, 'validation.log'))
    
    # Load config
    cfg = EasyDict(load_config(args.config))
    
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, args.checkpoint_pattern)))
    
    if not checkpoint_files:
        logger.error(f"No checkpoints found in {args.checkpoint_dir} with pattern {args.checkpoint_pattern}")
        return
    
    logger.info(f"Found {len(checkpoint_files)} checkpoints to validate")
    logger.info(f"Validating on datasets: {args.datasets}")
    
    # Initialize model architecture
    if args.model_name == 'NeuStereo':
        from NeuStereo.neustereo import NeuStereo
        base_model = NeuStereo(cfg).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    # Store all results
    all_results = []
    best_checkpoints = {dataset: {} for dataset in args.datasets}
    
    # Validate each checkpoint
    for checkpoint_path in tqdm(checkpoint_files, desc="Validating checkpoints"):
        checkpoint_name = os.path.basename(checkpoint_path)
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating checkpoint: {checkpoint_name}")
        
        # Load checkpoint
        model, epoch_num = load_checkpoint(checkpoint_path, base_model, device)
        if model is None:
            logger.error(f"Failed to load checkpoint: {checkpoint_path}")
            continue
        
        # Validate on each dataset
        results = validate_checkpoint(model, args.datasets, device)
        
        # Store results
        result_entry = {
            'checkpoint': checkpoint_name,
            'checkpoint_path': checkpoint_path,
            'epoch': epoch_num
        }
        
        # Flatten results for DataFrame
        for dataset, metrics in results.items():
            if metrics is not None:
                for metric, value in metrics.items():
                    result_entry[f"{dataset}_{metric}"] = value
                    
                    # Track best checkpoints
                    if dataset not in best_checkpoints:
                        best_checkpoints[dataset] = {}
                    if metric not in best_checkpoints[dataset]:
                        best_checkpoints[dataset][metric] = {
                            'value': float('inf') if 'error' in metric or metric in ['epe', 'd1'] else float('-inf'),
                            'checkpoint': None,
                            'epoch': None
                        }
                    
                    # Update best (lower is better for all metrics in stereo)
                    if value < best_checkpoints[dataset][metric]['value']:
                        best_checkpoints[dataset][metric]['value'] = value
                        best_checkpoints[dataset][metric]['checkpoint'] = checkpoint_name
                        best_checkpoints[dataset][metric]['epoch'] = epoch_num
        
        all_results.append(result_entry)
        
        # Log results for this checkpoint
        logger.info(f"Results for {checkpoint_name}:")
        for dataset in args.datasets:
            if results.get(dataset):
                metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in results[dataset].items()])
                logger.info(f"  {dataset}: {metrics_str}")
    
    # Save results to CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, 'validation_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")
    
    # Sort dataframe by epoch if available
    if 'epoch' in df.columns and df['epoch'].notna().any():
        df = df.sort_values('epoch')
    
    # Save sorted results
    sorted_csv_path = os.path.join(args.output_dir, 'validation_results_sorted.csv')
    df.to_csv(sorted_csv_path, index=False)
    
    # Print and save best checkpoints
    logger.info("\n" + "="*50)
    logger.info("BEST CHECKPOINTS PER DATASET AND METRIC:")
    logger.info("="*50)
    
    best_checkpoints_formatted = {}
    for dataset, metrics in best_checkpoints.items():
        logger.info(f"\n{dataset.upper()}:")
        best_checkpoints_formatted[dataset] = {}
        for metric, info in metrics.items():
            if info['checkpoint'] is not None:
                logger.info(f"  Best {metric}: {info['value']:.3f} "
                          f"(Checkpoint: {info['checkpoint']}, Epoch: {info['epoch']})")
                best_checkpoints_formatted[dataset][metric] = {
                    'value': float(info['value']),
                    'checkpoint': info['checkpoint'],
                    'epoch': info['epoch']
                }
    
    # Save best checkpoints to JSON
    json_path = os.path.join(args.output_dir, 'best_checkpoints.json')
    with open(json_path, 'w') as f:
        json.dump(best_checkpoints_formatted, f, indent=2)
    logger.info(f"\nBest checkpoints saved to {json_path}")
    
    # Create a summary table of best checkpoints
    summary_data = []
    for dataset in args.datasets:
        if dataset in best_checkpoints:
            for metric in best_checkpoints[dataset]:
                if best_checkpoints[dataset][metric]['checkpoint'] is not None:
                    summary_data.append({
                        'Dataset': dataset,
                        'Metric': metric,
                        'Best Value': best_checkpoints[dataset][metric]['value'],
                        'Checkpoint': best_checkpoints[dataset][metric]['checkpoint'],
                        'Epoch': best_checkpoints[dataset][metric]['epoch']
                    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(args.output_dir, 'best_checkpoints_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Summary table saved to {summary_csv_path}")
    
    # Create performance plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Plot performance over epochs for each metric
        if 'epoch' in df.columns and df['epoch'].notna().any():
            for dataset in args.datasets:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'{dataset.upper()} Performance Over Training', fontsize=14)
                
                metrics_to_plot = []
                if f'{dataset}_epe' in df.columns:
                    metrics_to_plot.append(('epe', f'{dataset}_epe', 'EPE'))
                if f'{dataset}_d1' in df.columns:
                    metrics_to_plot.append(('d1', f'{dataset}_d1', 'D1 (%)'))
                
                # Add the appropriate threshold metric
                for col in df.columns:
                    if dataset in col and 'error' in col:
                        metrics_to_plot.append(('thresh', col, col.split('_')[-1].upper()))
                        break
                
                for idx, (_, col, label) in enumerate(metrics_to_plot[:3]):
                    if col in df.columns:
                        axes[idx].plot(df['epoch'], df[col], 'o-')
                        axes[idx].set_xlabel('Epoch')
                        axes[idx].set_ylabel(label)
                        axes[idx].grid(True, alpha=0.3)
                        
                        # Mark the best point
                        best_idx = df[col].idxmin()
                        axes[idx].plot(df.loc[best_idx, 'epoch'], df.loc[best_idx, col], 
                                     'r*', markersize=15, label=f'Best: {df.loc[best_idx, col]:.3f}')
                        axes[idx].legend()
                
                plt.tight_layout()
                plot_path = os.path.join(args.output_dir, f'{dataset}_performance.png')
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                logger.info(f"Performance plot saved to {plot_path}")
                
    except ImportError:
        logger.info("Matplotlib not available, skipping plots")
    
    logger.info("\nValidation complete!")

if __name__ == "__main__":
    main()