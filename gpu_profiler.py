import torch
import functools
import time
import psutil
import os
import subprocess
from typing import Optional, Dict, Any
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class GPUMemoryProfiler:
    """
    A comprehensive GPU memory profiler for tracking memory usage in the RAG pipeline.
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.reset_stats()
        
    def reset_stats(self):
        """Reset all tracking statistics."""
        self.peak_memory = 0
        self.peak_reserved = 0
        self.memory_snapshots = []
        self.function_stats = {}
        
    def get_nvidia_smi_memory(self) -> Dict[str, float]:
        """Get GPU memory info from nvidia-smi (similar to what nvidia-smi shows)."""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total', 
                '--format=csv,noheader,nounits',
                f'--id={self.device_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                return {
                    "used_mb": used,
                    "total_mb": total,
                    "free_mb": total - used,
                    "utilization_percent": (used / total) * 100
                }
        except Exception as e:
            return {"error": f"nvidia-smi failed: {e}"}
        
        return {"error": "nvidia-smi not available"}
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information in MB."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        device = torch.cuda.device(self.device_id)
        
        # Get PyTorch memory info
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2  # MB
        free_pytorch = total - allocated
        
        # Update peak memory
        self.peak_memory = max(self.peak_memory, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        
        # Get nvidia-smi style memory info
        nvidia_smi_info = self.get_nvidia_smi_memory()
        
        result = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": total,
            "free_pytorch_mb": free_pytorch,
            "utilization_pytorch_percent": (allocated / total) * 100,
            "peak_allocated_mb": self.peak_memory,
            "peak_reserved_mb": self.peak_reserved
        }
        
        # Add nvidia-smi info if available
        if "error" not in nvidia_smi_info:
            result.update({
                "nvidia_smi_used_mb": nvidia_smi_info["used_mb"],
                "nvidia_smi_free_mb": nvidia_smi_info["free_mb"],
                "nvidia_smi_utilization_percent": nvidia_smi_info["utilization_percent"],
                "overhead_mb": nvidia_smi_info["used_mb"] - allocated  # Difference between nvidia-smi and PyTorch
            })
        
        return result
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system RAM information."""
        mem = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        process_mem = process.memory_info()
        
        return {
            "total_ram_gb": mem.total / 1024**3,
            "available_ram_gb": mem.available / 1024**3,
            "used_ram_gb": mem.used / 1024**3,
            "ram_percent": mem.percent,
            "process_ram_mb": process_mem.rss / 1024**2
        }
    
    def print_memory_summary(self, prefix: str = ""):
        """Print a formatted memory summary."""
        gpu_info = self.get_gpu_memory_info()
        sys_info = self.get_system_memory_info()
        
        print(f"\n{'='*70}")
        print(f"  {prefix} MEMORY SUMMARY")
        print(f"{'='*70}")
        
        if "error" not in gpu_info:
            print(f"GPU Memory (PyTorch perspective):")
            print(f"  ├── Allocated:    {gpu_info['allocated_mb']:.1f} MB")
            print(f"  ├── Reserved:     {gpu_info['reserved_mb']:.1f} MB") 
            print(f"  ├── Free:         {gpu_info['free_pytorch_mb']:.1f} MB")
            print(f"  ├── Total:        {gpu_info['total_mb']:.1f} MB")
            print(f"  ├── Usage:        {gpu_info['utilization_pytorch_percent']:.1f}%")
            print(f"  ├── Peak Alloc:   {gpu_info['peak_allocated_mb']:.1f} MB")
            print(f"  └── Peak Reserved: {gpu_info['peak_reserved_mb']:.1f} MB")
            
            # Add nvidia-smi perspective if available
            if "nvidia_smi_used_mb" in gpu_info:
                print(f"\nGPU Memory (nvidia-smi perspective):")
                print(f"  ├── Used:         {gpu_info['nvidia_smi_used_mb']:.1f} MB")
                print(f"  ├── Free:         {gpu_info['nvidia_smi_free_mb']:.1f} MB")
                print(f"  ├── Usage:        {gpu_info['nvidia_smi_utilization_percent']:.1f}%")
                print(f"  └── Overhead:     {gpu_info['overhead_mb']:.1f} MB")
                print(f"      (Difference between nvidia-smi and PyTorch allocated)")
        else:
            print(f"GPU: {gpu_info['error']}")
            
        print(f"\nSystem Memory:")
        print(f"  ├── Process RAM: {sys_info['process_ram_mb']:.1f} MB")
        print(f"  ├── System RAM:  {sys_info['ram_percent']:.1f}% used")
        print(f"  └── Available:   {sys_info['available_ram_gb']:.1f} GB")
        print(f"{'='*70}\n")
    
    def take_snapshot(self, label: str):
        """Take a memory snapshot with a label."""
        gpu_info = self.get_gpu_memory_info()
        sys_info = self.get_system_memory_info()
        
        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "gpu": gpu_info,
            "system": sys_info
        }
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def profile_function(self, func_name: str = None):
        """Decorator to profile GPU memory usage of a function."""
        def decorator(func):
            nonlocal func_name
            if func_name is None:
                func_name = func.__name__
                
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Take before snapshot
                before = self.take_snapshot(f"{func_name}_start")
                start_time = time.time()
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Take after snapshot
                    after = self.take_snapshot(f"{func_name}_end")
                    end_time = time.time()
                    
                    # Calculate stats
                    memory_diff = after["gpu"]["allocated_mb"] - before["gpu"]["allocated_mb"]
                    reserved_diff = after["gpu"]["reserved_mb"] - before["gpu"]["reserved_mb"]
                    execution_time = end_time - start_time
                    
                    # Calculate nvidia-smi difference if available
                    nvidia_smi_diff = 0
                    if "nvidia_smi_used_mb" in after["gpu"] and "nvidia_smi_used_mb" in before["gpu"]:
                        nvidia_smi_diff = after["gpu"]["nvidia_smi_used_mb"] - before["gpu"]["nvidia_smi_used_mb"]
                    
                    # Store function stats
                    self.function_stats[func_name] = {
                        "memory_change_mb": memory_diff,
                        "reserved_change_mb": reserved_diff,
                        "nvidia_smi_change_mb": nvidia_smi_diff,
                        "execution_time_s": execution_time,
                        "peak_memory_mb": after["gpu"]["peak_allocated_mb"],
                        "before_memory_mb": before["gpu"]["allocated_mb"],
                        "after_memory_mb": after["gpu"]["allocated_mb"]
                    }
                    
                    print(f"\n📊 {func_name} completed:")
                    print(f"   ⏱️  Execution time: {execution_time:.2f}s")
                    print(f"   🧠 PyTorch allocated: {memory_diff:+.1f} MB")
                    print(f"   🗄️  PyTorch reserved: {reserved_diff:+.1f} MB")
                    if nvidia_smi_diff != 0:
                        print(f"   🖥️  nvidia-smi used: {nvidia_smi_diff:+.1f} MB")
                    print(f"   📈 Memory after: {after['gpu']['allocated_mb']:.1f} MB")
                    
                    return result
                    
                except Exception as e:
                    # Take error snapshot
                    error = self.take_snapshot(f"{func_name}_error")
                    print(f"❌ Error in {func_name}: {str(e)}")
                    self.print_memory_summary(f"ERROR in {func_name}")
                    raise
                    
            return wrapper
        return decorator
    
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager to profile a code block."""
        before = self.take_snapshot(f"{block_name}_start")
        start_time = time.time()
        
        try:
            yield self
        finally:
            after = self.take_snapshot(f"{block_name}_end")
            end_time = time.time()
            
            memory_diff = after["gpu"]["allocated_mb"] - before["gpu"]["allocated_mb"]
            reserved_diff = after["gpu"]["reserved_mb"] - before["gpu"]["reserved_mb"]
            execution_time = end_time - start_time
            
            # Calculate nvidia-smi difference if available
            nvidia_smi_diff = 0
            if "nvidia_smi_used_mb" in after["gpu"] and "nvidia_smi_used_mb" in before["gpu"]:
                nvidia_smi_diff = after["gpu"]["nvidia_smi_used_mb"] - before["gpu"]["nvidia_smi_used_mb"]
            
            print(f"\n📊 {block_name} completed:")
            print(f"   ⏱️  Execution time: {execution_time:.2f}s")
            print(f"   🧠 PyTorch allocated: {memory_diff:+.1f} MB")
            print(f"   🗄️  PyTorch reserved: {reserved_diff:+.1f} MB")
            if nvidia_smi_diff != 0:
                print(f"   🖥️  nvidia-smi used: {nvidia_smi_diff:+.1f} MB")
            print(f"   📈 Memory after: {after['gpu']['allocated_mb']:.1f} MB")
    
    def print_function_summary(self):
        """Print summary of all profiled functions."""
        if not self.function_stats:
            print("No function profiling data available.")
            return
            
        print(f"\n{'='*90}")
        print(f"  FUNCTION PROFILING SUMMARY")
        print(f"{'='*90}")
        
        # Sort by memory usage
        sorted_funcs = sorted(
            self.function_stats.items(), 
            key=lambda x: abs(x[1]["memory_change_mb"]), 
            reverse=True
        )
        
        print(f"{'Function':<20} {'Time (s)':<8} {'PyTorch Δ (MB)':<15} {'Reserved Δ (MB)':<15} {'nvidia-smi Δ (MB)':<17} {'Peak (MB)':<10}")
        print(f"{'-'*88}")
        
        for func_name, stats in sorted_funcs:
            nvidia_smi_change = stats.get("nvidia_smi_change_mb", 0)
            print(f"{func_name:<20} "
                  f"{stats['execution_time_s']:<8.2f} "
                  f"{stats['memory_change_mb']:<+15.1f} "
                  f"{stats['reserved_change_mb']:<+15.1f} "
                  f"{nvidia_smi_change:<+17.1f} "
                  f"{stats['peak_memory_mb']:<10.1f}")
        
        total_memory_change = sum(stats["memory_change_mb"] for stats in self.function_stats.values())
        total_reserved_change = sum(stats["reserved_change_mb"] for stats in self.function_stats.values())
        total_nvidia_smi_change = sum(stats.get("nvidia_smi_change_mb", 0) for stats in self.function_stats.values())
        total_time = sum(stats["execution_time_s"] for stats in self.function_stats.values())
        
        print(f"{'-'*88}")
        print(f"{'TOTAL':<20} "
              f"{total_time:<8.2f} "
              f"{total_memory_change:<+15.1f} "
              f"{total_reserved_change:<+15.1f} "
              f"{total_nvidia_smi_change:<+17.1f} "
              f"{self.peak_memory:<10.1f}")
        print(f"{'='*90}\n")
        
        # Add explanation
        print("📝 Memory Explanation:")
        print("   PyTorch Δ:    Change in PyTorch allocated memory (tensors)")
        print("   Reserved Δ:   Change in PyTorch reserved memory (cache)")
        print("   nvidia-smi Δ: Change in total GPU memory used (what nvidia-smi shows)")
        print("   Peak:         Maximum PyTorch allocated memory during execution")
        print("\n💡 The difference between PyTorch allocated and nvidia-smi used includes:")
        print("   - CUDA context overhead")
        print("   - PyTorch memory cache (reserved but not allocated)")
        print("   - Other CUDA library memory usage")
    
    def save_profile_report(self, filename: str, timestamp_folder: str = None):
        """Save detailed profiling report to file."""
        import json
        
        # Create timestamped profiling folder if it doesn't exist
        if timestamp_folder is None:
            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        profile_dir = os.path.join("gpu_profiling_reports", timestamp_folder)
        os.makedirs(profile_dir, exist_ok=True)
        
        # Ensure filename is just the base name, add to profile directory
        base_filename = os.path.basename(filename)
        full_path = os.path.join(profile_dir, base_filename)
        
        report = {
            "peak_memory_mb": self.peak_memory,
            "peak_reserved_mb": self.peak_reserved,
            "function_stats": self.function_stats,
            "memory_snapshots": self.memory_snapshots,
            "device_info": {
                "device_id": self.device_id,
                "device_name": torch.cuda.get_device_name(self.device_id) if torch.cuda.is_available() else "N/A",
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
            }
        }
        
        with open(full_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Profile report saved to: {full_path}")
        return full_path
    
    def plot_memory_timeline(self, filename: str = None, show_plot: bool = False, timestamp_folder: str = None):
        """Plot GPU memory usage over time."""
        if not self.memory_snapshots:
            print("No memory snapshots available for plotting.")
            return None
            
        # Extract data from snapshots
        timestamps = []
        pytorch_allocated = []
        pytorch_reserved = []
        nvidia_smi_used = []
        labels = []
        
        start_time = self.memory_snapshots[0]["timestamp"]
        
        for snapshot in self.memory_snapshots:
            relative_time = snapshot["timestamp"] - start_time
            timestamps.append(relative_time)
            
            gpu_info = snapshot["gpu"]
            if "error" not in gpu_info:
                pytorch_allocated.append(gpu_info["allocated_mb"])
                pytorch_reserved.append(gpu_info["reserved_mb"])
                nvidia_smi_used.append(gpu_info.get("nvidia_smi_used_mb", 0))
            else:
                pytorch_allocated.append(0)
                pytorch_reserved.append(0)
                nvidia_smi_used.append(0)
            
            labels.append(snapshot["label"])
        
        # Create the plot
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Memory Usage Over Time
        ax1.plot(timestamps, pytorch_allocated, 'b-', linewidth=2, label='PyTorch Allocated', marker='o', markersize=4)
        ax1.plot(timestamps, pytorch_reserved, 'g--', linewidth=2, label='PyTorch Reserved', marker='s', markersize=4)
        if any(nvidia_smi_used):
            ax1.plot(timestamps, nvidia_smi_used, 'r:', linewidth=2, label='nvidia-smi Used', marker='^', markersize=4)
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('GPU Memory Usage Timeline', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Add function markers
        function_markers = []
        for i, label in enumerate(labels):
            if '_start' in label:
                function_name = label.replace('_start', '')
                ax1.axvline(x=timestamps[i], color='orange', linestyle='--', alpha=0.7)
                ax1.text(timestamps[i], max(pytorch_allocated) * 0.1, function_name, 
                        rotation=90, fontsize=8, ha='right', va='bottom')
        
        # Plot 2: Function Memory Changes
        if self.function_stats:
            functions = list(self.function_stats.keys())
            pytorch_changes = [stats["memory_change_mb"] for stats in self.function_stats.values()]
            reserved_changes = [stats["reserved_change_mb"] for stats in self.function_stats.values()]
            nvidia_changes = [stats.get("nvidia_smi_change_mb", 0) for stats in self.function_stats.values()]
            
            x = np.arange(len(functions))
            width = 0.25
            
            bars1 = ax2.bar(x - width, pytorch_changes, width, label='PyTorch Allocated Δ', color='blue', alpha=0.7)
            bars2 = ax2.bar(x, reserved_changes, width, label='PyTorch Reserved Δ', color='green', alpha=0.7)
            bars3 = ax2.bar(x + width, nvidia_changes, width, label='nvidia-smi Used Δ', color='red', alpha=0.7)
            
            ax2.set_xlabel('Functions')
            ax2.set_ylabel('Memory Change (MB)')
            ax2.set_title('Memory Change by Function', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(functions, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels on bars
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if abs(height) > 50:  # Only label significant changes
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:+.0f}',
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=8, fontweight='bold')
            
            add_value_labels(bars1)
            add_value_labels(bars2)
            add_value_labels(bars3)
        
        # Add overall info
        total_allocated = max(pytorch_allocated) if pytorch_allocated else 0
        peak_memory = self.peak_memory
        
        fig.suptitle(f'GPU Memory Profiling Report\n'
                    f'Peak PyTorch Memory: {peak_memory:.1f} MB | '
                    f'Max nvidia-smi: {max(nvidia_smi_used) if nvidia_smi_used else 0:.1f} MB',
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for suptitle
        
        # Save the plot
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpu_memory_profile_{timestamp}.png"
        
        # Create timestamped profiling folder if it doesn't exist
        if timestamp_folder is None:
            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        profile_dir = os.path.join("gpu_profiling_reports", timestamp_folder)
        os.makedirs(profile_dir, exist_ok=True)
        
        # Ensure filename is just the base name, add to profile directory
        base_filename = os.path.basename(filename)
        full_path = os.path.join(profile_dir, base_filename)
        
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Memory timeline plot saved to: {full_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return full_path
    
    def plot_memory_breakdown(self, filename: str = None, show_plot: bool = False, timestamp_folder: str = None):
        """Plot memory breakdown and efficiency analysis."""
        if not self.function_stats:
            print("No function statistics available for plotting.")
            return None
        
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        functions = list(self.function_stats.keys())
        pytorch_changes = [stats["memory_change_mb"] for stats in self.function_stats.values()]
        execution_times = [stats["execution_time_s"] for stats in self.function_stats.values()]
        peak_memories = [stats["peak_memory_mb"] for stats in self.function_stats.values()]
        
        # Plot 1: Memory vs Time Efficiency
        colors = plt.cm.viridis(np.linspace(0, 1, len(functions)))
        scatter = ax1.scatter(execution_times, pytorch_changes, 
                            s=[p/10 for p in peak_memories], 
                            c=colors, alpha=0.7, edgecolors='black')
        
        for i, func in enumerate(functions):
            ax1.annotate(func, (execution_times[i], pytorch_changes[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_ylabel('Memory Change (MB)')
        ax1.set_title('Memory vs Time Efficiency\n(Bubble size = Peak Memory)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Memory Usage Distribution
        positive_changes = [max(0, m) for m in pytorch_changes]
        ax2.pie(positive_changes, labels=functions, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Memory Allocation Distribution\n(Positive Changes Only)', fontweight='bold')
        
        # Plot 3: Execution Time Distribution
        ax3.barh(functions, execution_times, color=colors, alpha=0.7)
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_title('Execution Time by Function', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add time labels
        for i, time in enumerate(execution_times):
            ax3.text(time + max(execution_times)*0.01, i, f'{time:.2f}s', 
                    va='center', fontsize=9)
        
        # Plot 4: Memory Efficiency (MB per second)
        efficiency = [abs(m)/t if t > 0 else 0 for m, t in zip(pytorch_changes, execution_times)]
        bars = ax4.bar(functions, efficiency, color=colors, alpha=0.7)
        ax4.set_ylabel('Memory Change per Second (MB/s)')
        ax4.set_title('Memory Allocation Rate', fontweight='bold')
        ax4.set_xticklabels(functions, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add efficiency labels
        for bar, eff in zip(bars, efficiency):
            if eff > max(efficiency)*0.05:  # Only label significant values
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{eff:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add overall summary
        total_time = sum(execution_times)
        total_memory = sum(positive_changes)
        avg_efficiency = total_memory / total_time if total_time > 0 else 0
        
        fig.suptitle(f'GPU Memory Analysis Dashboard\n'
                    f'Total Time: {total_time:.2f}s | Total Memory: {total_memory:.1f} MB | '
                    f'Avg Rate: {avg_efficiency:.1f} MB/s',
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for suptitle
        
        # Save the plot
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gpu_memory_breakdown_{timestamp}.png"
        
        # Create timestamped profiling folder if it doesn't exist
        if timestamp_folder is None:
            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        profile_dir = os.path.join("gpu_profiling_reports", timestamp_folder)
        os.makedirs(profile_dir, exist_ok=True)
        
        # Ensure filename is just the base name, add to profile directory
        base_filename = os.path.basename(filename)
        full_path = os.path.join(profile_dir, base_filename)
        
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Memory breakdown plot saved to: {full_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return full_path


# Global profiler instance
profiler = GPUMemoryProfiler()

# Convenience functions
def print_gpu_memory(prefix: str = "", enabled: bool = True):
    """Print current GPU memory usage."""
    if enabled:
        profiler.print_memory_summary(prefix)

def profile_function(func_name: str = None, enabled: bool = True):
    """Decorator for profiling functions."""
    def decorator(func):
        if enabled:
            return profiler.profile_function(func_name)(func)
        else:
            return func
    return decorator

def profile_block(block_name: str, enabled: bool = True):
    """Context manager for profiling code blocks."""
    if enabled:
        return profiler.profile_block(block_name)
    else:
        from contextlib import nullcontext
        return nullcontext()

def reset_profiler(enabled: bool = True):
    """Reset the global profiler."""
    if enabled:
        profiler.reset_stats()

def save_profile_report(filename: str, timestamp_folder: str = None, enabled: bool = True):
    """Save profiling report."""
    if enabled:
        return profiler.save_profile_report(filename, timestamp_folder)
    return None

def print_function_summary(enabled: bool = True):
    """Print function profiling summary."""
    if enabled:
        profiler.print_function_summary()

def plot_memory_timeline(filename: str = None, show_plot: bool = False, timestamp_folder: str = None):
    """Plot GPU memory usage timeline."""
    return profiler.plot_memory_timeline(filename, show_plot, timestamp_folder)

def plot_memory_breakdown(filename: str = None, show_plot: bool = False, timestamp_folder: str = None):
    """Plot memory breakdown and analysis."""
    return profiler.plot_memory_breakdown(filename, show_plot, timestamp_folder)

def save_all_plots(prefix: str = None, timestamp_folder: str = None, enabled: bool = True):
    """Save both timeline and breakdown plots."""
    if not enabled:
        return None, None
        
    import datetime
    if timestamp_folder is None:
        timestamp_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if prefix is None:
        prefix = "gpu_profile"
    
    timeline_file = f"{prefix}_timeline.png"
    breakdown_file = f"{prefix}_breakdown.png"
    
    timeline_path = plot_memory_timeline(timeline_file, timestamp_folder=timestamp_folder)
    breakdown_path = plot_memory_breakdown(breakdown_file, timestamp_folder=timestamp_folder)
    
    return timeline_path, breakdown_path 