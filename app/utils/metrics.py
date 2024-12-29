import time
from collections import deque
import numpy as np
from threading import Lock

class ModelMetrics:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.lock = Lock()
        self.total_requests = 0
        self.start_time = time.time()

    def update_metrics(self, processing_time: float):
        with self.lock:
            self.processing_times.append(processing_time)
            self.total_requests += 1

    def get_stats(self):
        with self.lock:
            if not self.processing_times:
                return {
                    "average_processing_time": 0,
                    "p95_processing_time": 0,
                    "p99_processing_time": 0,
                    "requests_per_second": 0,
                    "total_requests": 0
                }

            times = np.array(self.processing_times)
            elapsed_time = time.time() - self.start_time

            return {
                "average_processing_time": float(np.mean(times)),
                "p95_processing_time": float(np.percentile(times, 95)),
                "p99_processing_time": float(np.percentile(times, 99)),
                "requests_per_second": self.total_requests / elapsed_time,
                "total_requests": self.total_requests
            }
