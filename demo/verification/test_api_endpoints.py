import sys
import os
from pathlib import Path
import requests
import time
import json
import asyncio
import aiohttp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent.parent))

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        print("\nTesting health endpoint...")
        response = requests.get(f"{self.base_url}/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        print("Health endpoint test passed!")
        return data
    
    async def test_process_endpoint(self, text):
        """Test the text processing endpoint"""
        async with aiohttp.ClientSession() as session:
            payload = {"text": text, "max_length": 1024}
            start_time = time.time()
            async with session.post(
                f"{self.base_url}/api/v1/process",
                json=payload
            ) as response:
                duration = time.time() - start_time
                data = await response.json()
                return {
                    "status_code": response.status,
                    "duration": duration,
                    "response": data
                }
    
    async def test_batch_endpoint(self, texts):
        """Test the batch processing endpoint"""
        async with aiohttp.ClientSession() as session:
            payload = {"texts": texts, "max_length": 1024}
            start_time = time.time()
            async with session.post(
                f"{self.base_url}/api/v1/batch",
                json=payload
            ) as response:
                duration = time.time() - start_time
                data = await response.json()
                return {
                    "status_code": response.status,
                    "duration": duration,
                    "response": data
                }
    
    def test_metrics_endpoint(self):
        """Test the metrics endpoint"""
        print("\nTesting metrics endpoint...")
        response = requests.get(f"{self.base_url}/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "model_stats" in data
        print("Metrics endpoint test passed!")
        return data
    
    async def run_load_test(self, num_requests=100):
        """Run a load test on the API"""
        print(f"\nRunning load test with {num_requests} requests...")
        
        # Generate test data
        sample_texts = [
            f"Sample text {i} for testing the API endpoint"
            for i in range(num_requests)
        ]
        
        # Run concurrent requests
        tasks = []
        for text in sample_texts:
            tasks.append(self.test_process_endpoint(text))
        
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        durations = [r["duration"] for r in results]
        status_codes = [r["status_code"] for r in results]
        
        stats = {
            "mean_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "success_rate": sum(1 for s in status_codes if s == 200) / len(status_codes)
        }
        
        print("\nLoad Test Results:")
        print(f"Mean response time: {stats['mean_duration']:.4f}s")
        print(f"Min response time: {stats['min_duration']:.4f}s")
        print(f"Max response time: {stats['max_duration']:.4f}s")
        print(f"Success rate: {stats['success_rate']*100:.2f}%")
        
        return stats

async def main():
    tester = APITester()
    
    # Test health endpoint
    health_data = tester.test_health_endpoint()
    print(json.dumps(health_data, indent=2))
    
    # Test metrics endpoint
    metrics_data = tester.test_metrics_endpoint()
    print(json.dumps(metrics_data, indent=2))
    
    # Run load test
    load_test_stats = await tester.run_load_test(num_requests=100)
    
    # Save results
    results = {
        "health_check": health_data,
        "metrics": metrics_data,
        "load_test": load_test_stats
    }
    
    with open("api_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTest results saved to 'api_test_results.json'")

if __name__ == "__main__":
    asyncio.run(main())
