import sys
import os
import torch
import time
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.models.modern_bert import ModernBERT

def run_simple_inference():
    print("Running simple inference demonstration...")
    
    # Initialize model
    model = ModernBERT(
        vocab_size=50257,  # GPT-2 vocabulary size for demonstration
        max_seq_len=8192,
        dim=768,
        depth=12,
        num_heads=12
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create sample input
    sample_text = torch.randint(0, 50257, (1, 1024), device=device)
    
    # Warm-up run
    print("Performing warm-up run...")
    with torch.no_grad():
        _ = model(sample_text)
    
    # Measure inference time
    print("\nMeasuring inference time...")
    num_runs = 5
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            output = model(sample_text)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Run {i+1}: {times[-1]:.4f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage inference time: {avg_time:.4f} seconds")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Device: {device}")

if __name__ == "__main__":
    run_simple_inference()
