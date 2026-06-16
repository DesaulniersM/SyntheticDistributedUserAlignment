import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import argparse

"""
UNIFIED SWARM REGISTRATION REPORT GENERATOR
===========================================
This script provides a standardized visualization engine for benchmarking
Multi-View Registration (MVR) solvers. 

It automatically creates a subdirectory for each dataset to prevent 
overwriting plots when comparing different studies.
"""

class SwarmReportGenerator:
    def __init__(self, data_path, output_base="results/plots", threshold=0.1):
        self.data_path = data_path
        # Create a unique folder for this dataset based on its filename
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]
        self.output_dir = os.path.join(output_base, dataset_name)
        self.threshold = threshold
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data not found at {self.data_path}")
        return pd.read_csv(self.data_path)

    def generate_all(self):
        """Generates the full suite of standardized plots."""
        solvers = self.df['solver'].unique()
        print(f"Dataset: {self.data_path} | Output: {self.output_dir}")
        for solver in solvers:
            print(f"  Generating reports for {solver}...")
            self.plot_success_heatmap(solver)
            self.plot_time_heatmap(solver)
        
        self.plot_reliability_frontier(solvers)

    def plot_success_heatmap(self, solver):
        """Visualizes probability of successful convergence (RMSE < threshold)."""
        df_s = self.df[self.df['solver'] == solver].copy()
        df_s['success'] = df_s['mean_rmse'] < self.threshold
        
        pivot = df_s.pivot_table(index='k_neighbors', columns='n_users', values='success', aggfunc='mean')
        pivot = pivot.sort_index(ascending=False)

        plt.figure(figsize=(10, 8))
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        im = plt.imshow(pivot, cmap='RdYlGn', norm=norm, aspect='auto')
        
        # Annotate with percentages
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val*100:.0f}%", ha="center", va="center", 
                             color="white" if (val < 0.3 or val > 0.7) else "black", fontweight='bold')

        plt.colorbar(im, label=f'Success Rate (RMSE < {self.threshold}m)')
        self._format_plot(pivot, f"{solver} Success Probability (N={len(df_s['n_users'].unique())})")
        
        save_path = f"{self.output_dir}/{solver.lower()}_success_map.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_time_heatmap(self, solver):
        """Visualizes computational cost (Average Duration)."""
        df_s = self.df[self.df['solver'] == solver]
        pivot = df_s.pivot_table(index='k_neighbors', columns='n_users', values='duration', aggfunc='mean')
        pivot = pivot.sort_index(ascending=False)

        plt.figure(figsize=(10, 8))
        max_t = self.df['duration'].quantile(0.95)
        im = plt.imshow(pivot, cmap='plasma', vmin=0, vmax=max_t, aspect='auto')
        
        # Annotate with seconds
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.1f}s", ha="center", va="center", color="white", fontsize=8)

        plt.colorbar(im, label='Average Duration (seconds)')
        self._format_plot(pivot, f"{solver} Solve Time Scaling")
        
        save_path = f"{self.output_dir}/{solver.lower()}_time_map.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_reliability_frontier(self, solvers):
        """Overlays the 90% success line for all solvers on a single chart."""
        plt.figure(figsize=(10, 7))
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, solver in enumerate(solvers):
            df_s = self.df[self.df['solver'] == solver].copy()
            df_s['success'] = df_s['mean_rmse'] < self.threshold
            
            pivot = df_s.pivot_table(index='k_neighbors', columns='n_users', values='success', aggfunc='mean')
            
            frontier_n = []
            frontier_k = []
            
            for n in pivot.columns:
                successful_ks = pivot.index[pivot[n] >= 0.90]
                if not successful_ks.empty:
                    frontier_n.append(n)
                    frontier_k.append(successful_ks.min())
            
            if frontier_n:
                plt.plot(frontier_n, frontier_k, 'o-', label=f"{solver} (90% Reliability)", 
                         color=colors_list[idx % len(colors_list)], linewidth=3, markersize=8)

        plt.title(f"Reliability Frontier (RMSE < {self.threshold}m)", fontsize=14)
        plt.xlabel("Number of Users (N)", fontsize=12)
        plt.ylabel("Required Neighbors (K)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylim(0, 11)
        
        save_path = f"{self.output_dir}/reliability_comparison.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _format_plot(self, pivot, title):
        plt.title(title, fontsize=14)
        plt.xlabel("Number of Users (N)", fontsize=12)
        plt.ylabel("Neighbors (K)", fontsize=12)
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.tight_layout()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Swarm Performance Reports")
    parser.add_argument("csv_path", help="Path to the experiment CSV file")
    parser.add_argument("--threshold", type=float, default=0.1, help="RMSE threshold for success (meters)")
    args = parser.parse_args()

    gen = SwarmReportGenerator(args.csv_path, threshold=args.threshold)
    gen.generate_all()
