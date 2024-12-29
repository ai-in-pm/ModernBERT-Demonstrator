# ModernBERT Demonstrator - Verification and Examples

This directory contains comprehensive demonstrations, benchmarks, and verification scripts to validate the ModernBERT implementation.

## Directory Structure

```
demo/
├── examples/
│   └── simple_inference.py      # Basic inference example
├── benchmarks/
│   └── attention_patterns.py    # Performance comparison of attention patterns
└── verification/
    └── test_api_endpoints.py    # API endpoint testing and validation
```

## Running the Demonstrations

### 1. Simple Inference Example

This script demonstrates basic model inference and provides performance metrics:

```bash
python demo/examples/simple_inference.py
```

Expected output:
- Inference time measurements
- Model parameter count
- Output shape verification
- Device utilization

### 2. Attention Pattern Benchmarks

Compare the performance of global and local attention patterns:

```bash
python demo/benchmarks/attention_patterns.py
```

This will generate:
- Performance measurements across different sequence lengths
- Comparison of global vs local attention
- Visual plot saved as 'attention_benchmark.png'

### 3. API Endpoint Verification

Comprehensive testing of the API endpoints:

```bash
python demo/verification/test_api_endpoints.py
```

Tests include:
- Health endpoint validation
- Metrics endpoint verification
- Load testing (100 concurrent requests)
- Response time measurements
- Success rate calculation

Results are saved to 'api_test_results.json'

## Verification Checklist

- [ ] Model loads successfully
- [ ] Inference runs on both CPU and GPU
- [ ] Attention patterns show expected performance characteristics
- [ ] API endpoints respond correctly
- [ ] System handles concurrent requests
- [ ] Memory usage remains within expected bounds

## Expected Results

1. **Inference Performance**:
   - Sub-second processing for 1024-token sequences
   - Linear scaling with sequence length
   - GPU utilization > 80%

2. **Attention Patterns**:
   - Local attention faster than global for long sequences
   - Memory usage scales efficiently with sequence length
   - Consistent performance across batch sizes

3. **API Performance**:
   - Response times < 100ms for basic requests
   - >95% success rate under load
   - Stable performance with concurrent requests

## Troubleshooting

If you encounter issues:

1. Check GPU availability and CUDA version
2. Verify environment variables in `.env`
3. Monitor memory usage during benchmarks
4. Check API server logs for errors

## Additional Notes

- All benchmarks should be run on a dedicated machine
- GPU tests require CUDA-compatible hardware
- API tests assume the server is running locally
- Results may vary based on hardware configuration
