from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import time
from prometheus_client import Counter, Histogram, generate_latest
import uvicorn

from app.models.modern_bert import ModernBERT
from app.utils.metrics import ModelMetrics

app = FastAPI(
    title="ModernBERT Demonstrator",
    description="API service demonstrating key innovations from the ModernBERT paper",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize metrics
REQUESTS = Counter('modernbert_requests_total', 'Total requests processed')
PROCESSING_TIME = Histogram('modernbert_processing_seconds', 'Time spent processing requests')

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModernBERT().to(device)
model.eval()

# Initialize metrics tracker
metrics_tracker = ModelMetrics()

class ProcessRequest(BaseModel):
    text: str
    max_length: Optional[int] = 8192
    temperature: Optional[float] = 1.0

class BatchRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 8192

@app.get("/")
async def root():
    return {"message": "ModernBERT Demonstrator API"}

@app.post("/api/v1/process")
async def process_text(request: ProcessRequest, background_tasks: BackgroundTasks):
    REQUESTS.inc()
    start_time = time.time()
    
    try:
        # Process text (simplified for demonstration)
        with torch.no_grad():
            # In a real implementation, you would:
            # 1. Tokenize the input text
            # 2. Process through the model
            # 3. Decode the output
            result = {"status": "processed", "length": len(request.text)}
        
        processing_time = time.time() - start_time
        PROCESSING_TIME.observe(processing_time)
        
        # Update metrics in background
        background_tasks.add_task(metrics_tracker.update_metrics, processing_time)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch")
async def batch_process(request: BatchRequest):
    REQUESTS.inc()
    start_time = time.time()
    
    try:
        results = []
        for text in request.texts:
            # Process each text (simplified for demonstration)
            results.append({"status": "processed", "length": len(text)})
        
        processing_time = time.time() - start_time
        PROCESSING_TIME.observe(processing_time)
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics")
async def get_metrics():
    return {
        "model_stats": metrics_tracker.get_stats(),
        "prometheus_metrics": generate_latest().decode()
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "max_sequence_length": model.max_seq_len,
        "parameter_count": model.get_num_params()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
