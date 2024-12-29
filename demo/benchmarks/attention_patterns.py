import sys
import os
import torch
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.models.modern_bert import AttentionBlock

def benchmark_attention_patterns():
    print("Benchmarking attention patterns...")
    
    # Test parameters
    batch_sizes = [1, 4, 8]
    sequence_lengths = [512, 1024, 2048, 4096, 8192]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize attention block
    attn = AttentionBlock(
        dim=768,
        num_heads=12,
        head_dim=64,
        window_size=256
    ).to(device)
    
    results = {
        'global': {(b, s): [] for b in batch_sizes for s in sequence_lengths},
        'local': {(b, s): [] for b in batch_sizes for s in sequence_lengths}
    }
    
    print("\nRunning benchmarks...")
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            print(f"\nTesting batch_size={batch_size}, sequence_length={seq_len}")
            
            # Create input tensor
            x = torch.randn(batch_size, seq_len, 768, device=device)
            
            # Warm-up
            with torch.no_grad():
                _ = attn(x, is_global=True)
                _ = attn(x, is_global=False)
            
            # Test global attention
            times = []
            with torch.no_grad():
                for _ in range(5):
                    start = time.time()
                    _ = attn(x, is_global=True)
                    torch.cuda.synchronize()
                    times.append(time.time() - start)
            results['global'][(batch_size, seq_len)] = np.mean(times)
            print(f"Global attention: {results['global'][(batch_size, seq_len)]:.4f}s")
            
            # Test local attention
            times = []
            with torch.no_grad():
                for _ in range(5):
                    start = time.time()
                    _ = attn(x, is_global=False)
                    torch.cuda.synchronize()
                    times.append(time.time() - start)
            results['local'][(batch_size, seq_len)] = np.mean(times)
            print(f"Local attention: {results['local'][(batch_size, seq_len)]:.4f}s")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for batch_size in batch_sizes:
        global_times = [results['global'][(batch_size, s)] for s in sequence_lengths]
        local_times = [results['local'][(batch_size, s)] for s in sequence_lengths]
        
        plt.plot(sequence_lengths, global_times, '-o', label=f'Global (batch={batch_size})')
        plt.plot(sequence_lengths, local_times, '--o', label=f'Local (batch={batch_size})')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Pattern Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('attention_benchmark.png')
    print("\nBenchmark results saved as 'attention_benchmark.png'")

if __name__ == "__main__":
    benchmark_attention_patterns()
