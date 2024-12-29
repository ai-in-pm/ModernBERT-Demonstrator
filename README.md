# ModernBERT Demonstrator

A FastAPI-based service demonstrating key innovations from the ModernBERT paper, including bidirectional encoding, extended context length, and alternating global/local attention patterns.

## Features

- 8,192 token context length support
- Alternating global/local attention implementation
- Memory-efficient processing
- Real-time inference benchmarking
- Streaming support for long contexts
- Hardware-aware optimizations

## Requirements

- Python 3.9+
- CUDA-compatible GPU
- Docker (optional)
- PostgreSQL
- Redis
- RabbitMQ

## Installation

1. Clone the repository
2. Copy the environment configuration:
   ```bash
   cp .env.example .env
   ```
3. Update the `.env` file with your configuration
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the API server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

- `/api/v1/process`: Process text with ModernBERT
- `/api/v1/batch`: Batch processing endpoint
- `/api/v1/stream`: Streaming endpoint for long contexts
- `/api/v1/metrics`: Performance metrics
- `/api/v1/health`: Model and service health status

## Architecture

The service implements:
- GeGLU activation function
- Rotary positional embeddings (RoPE)
- Complete model unpadding
- Global and local attention patterns
- Hardware-aware optimizations

## Performance

- 2x faster processing compared to standard BERT
- Optimized GPU utilization
- Efficient memory management
- Support for variable length sequences
- Large batch processing capabilities

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
