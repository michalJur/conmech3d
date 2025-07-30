# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# def convert_tensorboard_to_csv(logdir, output_file):
#     event_acc = EventAccumulator(logdir)
#     event_acc.Reload()
    
#     # Get scalar data
#     scalar_keys = event_acc.Tags()['scalars']
    
#     import pandas as pd
#     data = []
#     for key in scalar_keys:
#         scalar_events = event_acc.Scalars(key)
#         for event in scalar_events:
#             data.append({
#                 'metric': key,
#                 'step': event.step,
#                 'value': event.value,
#                 'wall_time': event.wall_time
#             })
    
#     df = pd.DataFrame(data)
#     df.to_csv(output_file, index=False)
#     return df

# # Convert and then import to other tools

# df = convert_tensorboard_to_csv('/home/mjureczka/Desktop/conmech3d/log', 'metrics.csv')

# # Now you can subsample as needed
# # sampled_df = df[df['step'] % 10 == 0]  # Every 10th point


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Please install tensorboard: pip install tensorboard")
    exit(1)

def convert_tensorboard_to_csv(logdir, output_file):
    """Convert TensorBoard logs to CSV format"""
    try:
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()
        
        # Get scalar data
        scalar_keys = event_acc.Tags()['scalars']
        
        if not scalar_keys:
            print(f"No scalar data found in {logdir}")
            return None
        
        data = []
        for key in scalar_keys:
            scalar_events = event_acc.Scalars(key)
            for event in scalar_events:
                data.append({
                    'metric': key,
                    'step': event.step,
                    'value': event.value,
                    'wall_time': event.wall_time
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Converted {logdir} -> {output_file} ({len(df)} records)")
        return df
        
    except Exception as e:
        print(f"Error processing {logdir}: {str(e)}")
        return None

def process_all_log_folders(base_log_dir="log", output_dir="converted_logs"):
    """Process all subfolders in the log directory"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    base_path = Path(base_log_dir)
    if not base_path.exists():
        print(f"Log directory '{base_log_dir}' does not exist!")
        return []
    
    converted_files = []
    
    # Find all subdirectories in the log folder
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            # Check if it contains TensorBoard event files
            event_files = list(subfolder.glob("events.out.tfevents.*"))
            
            if event_files:
                output_file = Path(output_dir) / f"{subfolder.name}.csv"
                df = convert_tensorboard_to_csv(str(subfolder), str(output_file))
                
                if df is not None:
                    converted_files.append({
                        'name': subfolder.name,
                        'file': str(output_file),
                        'df': df
                    })
            else:
                print(f"No TensorBoard event files found in {subfolder}")
    
    return converted_files

def plot_all_metrics(converted_files, save_plots=True, plot_dir="plots"):
    """Load all CSV files and create plots for each metric"""
    
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    if not converted_files:
        print("No converted files to plot!")
        return
    
    # Collect all unique metrics across all files
    all_metrics = set()
    for file_info in converted_files:
        df = file_info['df']
        all_metrics.update(df['metric'].unique())
    
    print(f"Found metrics: {sorted(all_metrics)}")
    
    # Create plots for each metric
    for metric in sorted(all_metrics):
        plt.figure(figsize=(12, 8))
        
        # Plot data from each experiment/subfolder
        for file_info in converted_files:
            df = file_info['df']
            metric_data = df[df['metric'] == metric]
            
            if not metric_data.empty:
                plt.plot(metric_data['step'], metric_data['value'], 
                        label=file_info['name'], marker='o', markersize=2, alpha=0.8)
        
        plt.title(f'Metric: {metric}', fontsize=14, fontweight='bold')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            # Clean metric name for filename
            safe_metric_name = metric.replace('/', '_').replace('\\', '_')
            plot_file = Path(plot_dir) / f"{safe_metric_name}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {plot_file}")
        
        plt.show()

def plot_summary_dashboard(converted_files, save_plot=True, plot_dir="plots"):
    """Create a summary dashboard with subplots for main metrics"""
    
    if not converted_files:
        return
    
    # Get common metrics (present in most experiments)
    metric_counts = {}
    for file_info in converted_files:
        df = file_info['df']
        for metric in df['metric'].unique():
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
    
    # Select top metrics by frequency
    common_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    common_metrics = [metric for metric, count in common_metrics]
    
    if not common_metrics:
        print("No common metrics found for dashboard")
        return
    
    # Create subplot dashboard
    n_metrics = len(common_metrics)
    cols = 3
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, metric in enumerate(common_metrics):
        ax = axes[idx]
        
        for file_info in converted_files:
            df = file_info['df']
            metric_data = df[df['metric'] == metric]
            
            if not metric_data.empty:
                ax.plot(metric_data['step'], metric_data['value'], 
                       label=file_info['name'], alpha=0.8)
        
        ax.set_title(metric, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs(plot_dir, exist_ok=True)
        dashboard_file = Path(plot_dir) / "dashboard_summary.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"Saved dashboard: {dashboard_file}")
    
    plt.show()

def main():
    log_dir = '/home/mjureczka/Desktop/conmech3d/log'
    """Main execution function"""
    print("Starting TensorBoard log conversion and plotting...")
    
    # Step 1: Convert all TensorBoard logs to CSV
    converted_files = process_all_log_folders(
        base_log_dir=log_dir, 
        output_dir="output/converted_logs"
    )
    
    if not converted_files:
        print("No TensorBoard logs found or converted!")
        return
    
    print(f"\nSuccessfully converted {len(converted_files)} log folders:")
    for file_info in converted_files:
        print(f"  - {file_info['name']}: {len(file_info['df'])} records")
    
    # Step 2: Create individual plots for each metric
    print("\nCreating individual metric plots...")
    plot_all_metrics(converted_files, save_plots=True, plot_dir="plots")
    
    # Step 3: Create summary dashboard
    print("\nCreating summary dashboard...")
    plot_summary_dashboard(converted_files, save_plot=True, plot_dir="plots")
    
    print("\nAll done! Check the 'converted_logs' and 'plots' directories for outputs.")

if __name__ == "__main__":
    main()