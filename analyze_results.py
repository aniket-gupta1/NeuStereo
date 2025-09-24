import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_path):
    """Load validation results from CSV"""
    return pd.read_csv(results_path)

def analyze_convergence(df, dataset, metric, window=5):
    """Analyze when the model converged for a specific metric"""
    col_name = f"{dataset}_{metric}"
    if col_name not in df.columns:
        return None
    
    # Sort by epoch
    df_sorted = df.sort_values('epoch')
    
    # Calculate rolling mean and std
    rolling_mean = df_sorted[col_name].rolling(window=window, center=True).mean()
    rolling_std = df_sorted[col_name].rolling(window=window, center=True).std()
    
    # Find the epoch where performance stabilizes (low std for window epochs)
    threshold = rolling_std.quantile(0.2)  # Bottom 20% of variation
    stable_epochs = df_sorted[rolling_std < threshold]['epoch'].values
    
    if len(stable_epochs) > 0:
        convergence_epoch = stable_epochs[0]
    else:
        convergence_epoch = None
    
    return {
        'convergence_epoch': convergence_epoch,
        'best_value': df_sorted[col_name].min(),
        'best_epoch': df_sorted.loc[df_sorted[col_name].idxmin(), 'epoch'],
        'final_value': df_sorted[col_name].iloc[-1],
        'final_epoch': df_sorted['epoch'].iloc[-1]
    }

def compare_datasets(df):
    """Compare performance across datasets"""
    comparison = {}
    
    datasets = ['eth3d', 'middlebury', 'kitti', 'flyingthings']
    
    # Define dataset-specific metrics
    dataset_metrics = {
        'eth3d': ['epe', '1px_error'],
        'middlebury': ['epe', '2px_error'],
        'kitti': ['epe', '3px_error'],
        'flyingthings': ['epe', '1px_error']
    }
    
    # First handle EPE (common across all)
    comparison['epe'] = {}
    for dataset in datasets:
        col_name = f"{dataset}_epe"
        if col_name in df.columns:
            comparison['epe'][dataset] = {
                'mean': df[col_name].mean(),
                'std': df[col_name].std(),
                'min': df[col_name].min(),
                'max': df[col_name].max(),
                'best_epoch': df.loc[df[col_name].idxmin(), 'epoch'] if 'epoch' in df.columns else None
            }
    
    # Handle pixel errors
    comparison['pixel_error'] = {}
    for dataset, metrics in dataset_metrics.items():
        for metric in metrics:
            if 'px_error' in metric:
                col_name = f"{dataset}_{metric}"
                if col_name in df.columns:
                    comparison['pixel_error'][f"{dataset}_{metric}"] = {
                        'mean': df[col_name].mean(),
                        'std': df[col_name].std(),
                        'min': df[col_name].min(),
                        'max': df[col_name].max(),
                        'best_epoch': df.loc[df[col_name].idxmin(), 'epoch'] if 'epoch' in df.columns else None
                    }
    
    return comparison

def find_robust_checkpoint(df, datasets=None, top_k=5):
    """Find checkpoints that perform well across multiple datasets/metrics"""
    
    if datasets is None:
        # Automatically detect datasets
        datasets = []
        for col in df.columns:
            if '_epe' in col:
                datasets.append(col.replace('_epe', ''))
        datasets = list(set(datasets))
    
    # Define dataset-specific metrics to use for robustness scoring
    dataset_metrics = {
        'eth3d': ['epe', '1px_error'],
        'middlebury': ['epe', '2px_error'],
        'kitti': ['epe', '3px_error'],
        'flyingthings': ['epe', '1px_error']
    }
    
    # Calculate normalized scores for each checkpoint
    scores = pd.DataFrame(index=df.index)
    
    for dataset in datasets:
        if dataset in dataset_metrics:
            for metric in dataset_metrics[dataset]:
                col_name = f"{dataset}_{metric}"
                if col_name in df.columns:
                    # Normalize (lower is better, so we invert)
                    min_val = df[col_name].min()
                    max_val = df[col_name].max()
                    if max_val > min_val:
                        scores[col_name] = 1 - (df[col_name] - min_val) / (max_val - min_val)
                    else:
                        scores[col_name] = 1.0
    
    # Calculate mean score across all metrics
    df['robustness_score'] = scores.mean(axis=1)
    
    # Get top k checkpoints
    top_checkpoints = df.nlargest(top_k, 'robustness_score')[['checkpoint', 'epoch', 'robustness_score']]
    
    # Add individual metric values
    for dataset in datasets:
        if dataset in dataset_metrics:
            for metric in dataset_metrics[dataset]:
                col_name = f"{dataset}_{metric}"
                if col_name in df.columns:
                    top_checkpoints[col_name] = df.loc[top_checkpoints.index, col_name]
    
    return top_checkpoints

def plot_comprehensive_analysis(df, output_dir):
    """Create comprehensive analysis plots"""
    
    # Detect available datasets and metrics
    datasets = []
    for col in df.columns:
        if '_epe' in col:
            datasets.append(col.replace('_epe', ''))
    datasets = list(set(datasets))
    
    if not datasets:
        print("No datasets found in results")
        return
    
    # Create subplots - now just 2 columns (EPE and Pixel Error)
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(12, 5*n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for idx, dataset in enumerate(datasets):
        # EPE plot
        if f'{dataset}_epe' in df.columns:
            ax = axes[idx, 0]
            df_sorted = df.sort_values('epoch')
            ax.plot(df_sorted['epoch'], df_sorted[f'{dataset}_epe'], 'o-', alpha=0.7)
            ax.set_title(f'{dataset.upper()} - EPE')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('EPE')
            ax.grid(True, alpha=0.3)
            
            # Mark best
            best_idx = df[f'{dataset}_epe'].idxmin()
            ax.axhline(y=df.loc[best_idx, f'{dataset}_epe'], color='r', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f'Best: {df.loc[best_idx, f"{dataset}_epe"]:.3f} @ Epoch {df.loc[best_idx, "epoch"]}',
                   transform=ax.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Pixel error plot
        px_error_col = None
        for col in df.columns:
            if dataset in col and 'px_error' in col:
                px_error_col = col
                break
        
        if px_error_col:
            ax = axes[idx, 1]
            df_sorted = df.sort_values('epoch')
            ax.plot(df_sorted['epoch'], df_sorted[px_error_col], 'o-', alpha=0.7, color='green')
            
            # Extract the pixel threshold from column name (1px, 2px, or 3px)
            px_threshold = px_error_col.split('_')[-1].upper()
            ax.set_title(f'{dataset.upper()} - {px_threshold} Error')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Error (%)')
            ax.grid(True, alpha=0.3)
            
            # Mark best
            best_idx = df[px_error_col].idxmin()
            ax.axhline(y=df.loc[best_idx, px_error_col], color='r', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f'Best: {df.loc[best_idx, px_error_col]:.2f}% @ Epoch {df.loc[best_idx, "epoch"]}',
                   transform=ax.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Training Progress Analysis (EPE & Pixel Error)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap - now excluding d1 metrics
    metrics_cols = [col for col in df.columns if any(x in col for x in ['_epe', '_error']) and '_d1' not in col]
    if len(metrics_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[metrics_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Metrics Correlation Matrix (EPE & Pixel Errors)', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze validation results')
    parser.add_argument('--results_csv', type=str, required=True,
                        help='Path to validation results CSV')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis results')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top robust checkpoints to show')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(args.results_csv).parent
    
    # Load results
    df = load_results(args.results_csv)
    print(f"Loaded {len(df)} checkpoint results")
    
    # Analyze convergence
    print("\n" + "="*50)
    print("CONVERGENCE ANALYSIS")
    print("="*50)
    
    # Define dataset-specific pixel error metrics
    dataset_pixel_metrics = {
        'eth3d': '1px_error',
        'middlebury': '2px_error',
        'kitti': '3px_error'
    }
    
    datasets = ['eth3d', 'middlebury', 'kitti']
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        
        # EPE convergence
        if f'{dataset}_epe' in df.columns:
            convergence_epe = analyze_convergence(df, dataset, 'epe')
            if convergence_epe:
                print(f"  EPE:")
                print(f"    Best: {convergence_epe['best_value']:.3f} at epoch {convergence_epe['best_epoch']}")
                print(f"    Final: {convergence_epe['final_value']:.3f} at epoch {convergence_epe['final_epoch']}")
                if convergence_epe['convergence_epoch']:
                    print(f"    Converged around epoch: {convergence_epe['convergence_epoch']}")
        
        # Pixel error convergence
        if dataset in dataset_pixel_metrics:
            px_metric = dataset_pixel_metrics[dataset]
            if f'{dataset}_{px_metric}' in df.columns:
                convergence_px = analyze_convergence(df, dataset, px_metric)
                if convergence_px:
                    print(f"  {px_metric.upper()}:")
                    print(f"    Best: {convergence_px['best_value']:.2f}% at epoch {convergence_px['best_epoch']}")
                    print(f"    Final: {convergence_px['final_value']:.2f}% at epoch {convergence_px['final_epoch']}")
                    if convergence_px['convergence_epoch']:
                        print(f"    Converged around epoch: {convergence_px['convergence_epoch']}")
    
    # Dataset comparison
    print("\n" + "="*50)
    print("DATASET COMPARISON")
    print("="*50)
    
    comparison = compare_datasets(df)
    
    # Print EPE comparison
    if 'epe' in comparison:
        print(f"\nEPE across datasets:")
        for dataset, stats in comparison['epe'].items():
            print(f"  {dataset}: min={stats['min']:.3f}, mean={stats['mean']:.3f}±{stats['std']:.3f}, best_epoch={stats['best_epoch']}")
    
    # Print pixel error comparison
    if 'pixel_error' in comparison:
        print(f"\nPixel Errors across datasets:")
        for metric_name, stats in comparison['pixel_error'].items():
            dataset = metric_name.split('_')[0]
            metric_type = '_'.join(metric_name.split('_')[1:])
            print(f"  {dataset} ({metric_type}): min={stats['min']:.2f}%, mean={stats['mean']:.2f}%±{stats['std']:.2f}%, best_epoch={stats['best_epoch']}")
    
    # Find robust checkpoints
    print("\n" + "="*50)
    print(f"TOP {args.top_k} MOST ROBUST CHECKPOINTS")
    print("="*50)
    print("(Based on EPE and dataset-specific pixel error metrics)")
    
    robust_checkpoints = find_robust_checkpoint(df, top_k=args.top_k)
    print("\nCheckpoints that perform well across all datasets:")
    
    # Format the output nicely
    display_cols = ['checkpoint', 'epoch', 'robustness_score']
    # Add EPE columns
    for dataset in ['eth3d', 'middlebury', 'kitti']:
        if f'{dataset}_epe' in robust_checkpoints.columns:
            display_cols.append(f'{dataset}_epe')
    # Add pixel error columns
    for dataset, px_metric in dataset_pixel_metrics.items():
        if f'{dataset}_{px_metric}' in robust_checkpoints.columns:
            display_cols.append(f'{dataset}_{px_metric}')
    
    print(robust_checkpoints[display_cols].to_string())
    
    # Save robust checkpoints
    robust_path = Path(args.output_dir) / 'robust_checkpoints.csv'
    robust_checkpoints.to_csv(robust_path, index=False)
    print(f"\nRobust checkpoints saved to: {robust_path}")
    
    # Generate plots
    print("\nGenerating analysis plots...")
    plot_comprehensive_analysis(df, args.output_dir)
    print(f"Plots saved to: {args.output_dir}")
    
    # Generate final report
    report = {
        'total_checkpoints': len(df),
        'datasets_evaluated': [d for d in datasets if f'{d}_epe' in df.columns],
        'scoring_metrics': {
            'eth3d': ['epe', '1px_error'],
            'middlebury': ['epe', '2px_error'],
            'kitti': ['epe', '3px_error']
        },
        'best_overall': robust_checkpoints.iloc[0].to_dict() if len(robust_checkpoints) > 0 else None,
        'convergence_analysis': {},
        'dataset_comparison': comparison
    }
    
    for dataset in datasets:
        report['convergence_analysis'][dataset] = {}
        if f'{dataset}_epe' in df.columns:
            report['convergence_analysis'][dataset]['epe'] = analyze_convergence(df, dataset, 'epe')
        if dataset in dataset_pixel_metrics:
            px_metric = dataset_pixel_metrics[dataset]
            if f'{dataset}_{px_metric}' in df.columns:
                report['convergence_analysis'][dataset][px_metric] = analyze_convergence(df, dataset, px_metric)
    
    report_path = Path(args.output_dir) / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nComplete analysis report saved to: {report_path}")
    
    # Print which metrics were used for robustness scoring
    print("\n" + "="*50)
    print("SCORING CRITERIA")
    print("="*50)
    print("Robustness scores were calculated using:")
    print("  - ETH3D: EPE + 1px error")
    print("  - Middlebury: EPE + 2px error")
    print("  - KITTI: EPE + 3px error")
    print("\nLower values are better for all metrics.")

if __name__ == "__main__":
    main()