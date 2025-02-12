import atexit
import threading
import time
from typing import List, Dict
import psutil
import logging
from memory_profiler import memory_usage
from datetime import datetime
import os
from pathlib import Path

# Set up a dedicated logger for profiling
logger = logging.getLogger(__name__)

def setup_profiling_logger():
    """Set up dedicated logging for profiling metrics."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a file handler
    log_file = log_dir / f"profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return log_file

class ApplicationProfiler:
    """Handles memory and CPU profiling for the application."""
    
    def __init__(self, sampling_interval: float = 5.0):
        self.log_file = setup_profiling_logger()
        logger.info(f"Profiling logs will be written to: {self.log_file}")
        
        self.sampling_interval = sampling_interval
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.start_time = datetime.now()
        self.process = psutil.Process()
        self._stop_sampling = threading.Event()
        self._sampling_thread = None
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        # Record baseline
        self.baseline_memory = self._get_memory_usage()
        self.baseline_cpu = self.process.cpu_percent()
        
        self._log_startup_metrics()
        
        # Start sampling thread
        self._start_sampling()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MiB."""
        return memory_usage(-1, interval=.1, timeout=1)[0]
    
    def _log_startup_metrics(self):
        """Log baseline metrics at startup."""
        logger.info("=== Application Profiling Started ===")
        logger.info(f"Baseline Memory Usage: {self.baseline_memory:.2f} MiB")
        logger.info(f"Baseline CPU Usage: {self.baseline_cpu:.1f}%")
        logger.info(f"Start Time: {self.start_time}")
    
    def _sampling_loop(self):
        """Background thread function to sample metrics."""
        while not self._stop_sampling.is_set():
            try:
                # Sample memory and CPU
                mem_usage = self._get_memory_usage()
                cpu_usage = self.process.cpu_percent()
                
                self.memory_samples.append(mem_usage)
                self.cpu_samples.append(cpu_usage)
                
                # Detailed logging if needed
                logger.debug(f"Memory: {mem_usage:.2f} MiB, CPU: {cpu_usage:.1f}%")
                
                # Wait for next sample
                self._stop_sampling.wait(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in profiling sample: {str(e)}")
    
    def _start_sampling(self):
        """Start the background sampling thread."""
        self._sampling_thread = threading.Thread(
            target=self._sampling_loop,
            name="ProfilingSampler",
            daemon=True
        )
        self._sampling_thread.start()
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current memory and CPU metrics."""
        return {
            'memory_mib': self._get_memory_usage(),
            'cpu_percent': self.process.cpu_percent(),
            'memory_diff': self._get_memory_usage() - self.baseline_memory
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics from collected samples."""
        if not self.memory_samples or not self.cpu_samples:
            return {}
            
        return {
            'avg_memory_mib': sum(self.memory_samples) / len(self.memory_samples),
            'max_memory_mib': max(self.memory_samples),
            'min_memory_mib': min(self.memory_samples),
            'avg_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples),
            'max_cpu_percent': max(self.cpu_samples),
            'memory_diff_mib': self.memory_samples[-1] - self.baseline_memory if self.memory_samples else 0
        }
    
    def shutdown(self):
        """Clean shutdown of profiler and log final statistics."""
        if self._stop_sampling.is_set():
            return
            
        self._stop_sampling.set()
        if self._sampling_thread:
            self._sampling_thread.join(timeout=2.0)
        
        # Get final metrics
        final_metrics = self.get_current_metrics()
        stats = self.get_statistics()
        runtime = datetime.now() - self.start_time
        
        # Log final report
        logger.info("\n=== Application Profiling Report ===")
        logger.info(f"Total Runtime: {runtime}")
        logger.info(f"Final Memory Usage: {final_metrics['memory_mib']:.2f} MiB")
        logger.info(f"Memory Change: {final_metrics['memory_diff']:.2f} MiB")
        logger.info(f"Average Memory Usage: {stats.get('avg_memory_mib', 0):.2f} MiB")
        logger.info(f"Peak Memory Usage: {stats.get('max_memory_mib', 0):.2f} MiB")
        logger.info(f"Average CPU Usage: {stats.get('avg_cpu_percent', 0):.1f}%")
        logger.info(f"Peak CPU Usage: {stats.get('max_cpu_percent', 0):.1f}%")
        logger.info("=== End Profiling Report ===\n") 