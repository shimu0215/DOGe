"""
Base experiment processor module
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from copy import deepcopy
import pickle
import time
import os
import multiprocessing as mp

from .cost_tracker import CostTracker
from exps_research.unified_framework import setup_model
from exps_research.unified_framework.utils import append_result

# Global counter for multiprocessing progress tracking
_progress_counter = None

# Top-level function for process pool to avoid pickling issues
def process_entry_in_process(
        entry: Dict, 
        worker_id: int, 
        model_kwargs: Dict[str, Any], 
        use_local_model: bool, 
        verbose_worker: bool, 
        processor_class_module: str, 
        processor_class_name: str,
        use_single_endpoint: bool = False,
        **kwargs
        ):
    """
    Top-level function to process a single entry in a separate process
    
    Args:
        entry: The entry to process
        worker_id: ID of the worker
        model_kwargs: Model configuration parameters
        use_local_model: Whether to use local model
        verbose_worker: Whether this worker should show verbose output
        processor_class_module: The module containing the processor class
        processor_class_name: The name of the processor class
        use_single_endpoint: Whether to use a single API endpoint for all workers
        **kwargs: Additional parameters to pass to process_entry
        
    Returns:
        Processed result
    """
    # Dynamically import the processor class
    import importlib
    module = importlib.import_module(processor_class_module)
    processor_class = getattr(module, processor_class_name)
    
    # Make a deep copy of model_kwargs to avoid modifying the original
    model_kwargs_copy = deepcopy(model_kwargs)
    
    # Configure model parameters for this worker
    if use_local_model:
        model_kwargs_copy["local_device_id"] = str(worker_id)
    else:
        # Only modify API base if it's not explicitly set
        if use_single_endpoint:
            model_kwargs_copy["api_base"] = "http://0.0.0.0:8000/v1"
        else:
            model_kwargs_copy["api_base"] = f"http://0.0.0.0:{8000 + worker_id}/v1"
    
    # Create model with the modified parameters
    model = setup_model(**model_kwargs_copy)
    
    # Create a temporary processor instance
    processor = processor_class(model_kwargs, **kwargs)
    
    # Process the entry
    result = processor.process_entry(
        entry, 
        model,
        verbose_worker=verbose_worker,
        **kwargs
    )
    
    return result


class ExperimentProcessor(ABC):
    """
    Base class for experiment processors
    
    This abstract class defines the interface and common functionality
    for all experiment processors. Concrete implementations should inherit
    from this class and implement the process_entry method.
    """
    
    def __init__(self, model_kwargs: Dict[str, Any], **kwargs):
        """
        Initialize experiment processor
        
        Args:
            model_kwargs: Model configuration parameters
            **kwargs: Additional experiment-specific parameters
        """
        self.model_kwargs = model_kwargs
        self.cost_tracker = CostTracker()
        self.track_cost = kwargs.get('track_cost', False)
        self.verbose = kwargs.get('verbose', False)
        
        # Set up cost tracking if enabled
        if self.track_cost:
            self.cost_tracker.reset(kwargs.get('cost_threshold'))
    
    @abstractmethod
    def process_entry(self, entry: Dict, model, **kwargs) -> Dict:
        """
        Process a single experiment entry
        
        Args:
            entry: Dictionary containing a question/problem
            model: Model instance to use
            **kwargs: Additional experiment-specific parameters
            
        Returns:
            Result dictionary with generated answer and metadata
        """
        pass
    
    def create_model(self, worker_id: int = 0, use_local_model: bool = False, use_single_endpoint: bool = False) -> Any:
        """
        Create a model instance for this worker
        
        Args:
            worker_id: ID of the worker thread/process
            use_local_model: Whether to use a local model
            use_single_endpoint: Whether to use a single API endpoint for all workers
            
        Returns:
            Model instance
        """
        model_kwargs = deepcopy(self.model_kwargs)
        
        if use_local_model:
            model_kwargs["local_device_id"] = str(worker_id)
        else:
            # Only modify API base if it's not explicitly set
            if not model_kwargs.get('api_base'):
                if use_single_endpoint:
                    model_kwargs["api_base"] = "http://0.0.0.0:8000/v1"
                else:
                    model_kwargs["api_base"] = f"http://0.0.0.0:{8000 + worker_id}/v1"
        
        return setup_model(**model_kwargs)
    
    def create_models(self, max_workers: int, use_local_model: bool = False, use_single_endpoint: bool = False) -> List:
        """
        Create model instances for all workers
        
        Args:
            max_workers: Number of worker threads/processes
            use_local_model: Whether to use local models
            use_single_endpoint: Whether to use a single API endpoint for all workers
            
        Returns:
            List of model instances
        """
        return [self.create_model(i, use_local_model, use_single_endpoint) for i in range(max_workers)]
    
    def process_dataset(
        self,
        entries: List[Dict],
        output_file: Optional[str] = None,
        max_workers: int = 1,
        debug: bool = False,
        use_local_model: bool = False,
        use_process_pool: bool = True,  # Default to process pool for reliable timeouts
        use_single_endpoint: bool = False,  # Use a single API endpoint for all workers
        per_task_timeout: int = 300,  # Max seconds per question before forcibly skipping
        **kwargs
    ) -> List[Dict]:
        """
        Process a dataset of entries
        
        Args:
            entries: List of dataset entries
            output_file: Path to output file
            max_workers: Maximum number of concurrent workers
            debug: Whether to run in debug mode
            use_local_model: Whether to use local models
            use_process_pool: Whether to use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
                              ProcessPoolExecutor is recommended for reliable timeouts in Python code execution
            use_single_endpoint: Whether to use a single API endpoint (port 8000) for all workers
            **kwargs: Additional experiment-specific parameters
            
        Returns:
            List of processed results
        """
        results = []
        
        # Limit entries in debug mode
        if debug:
            entries = entries[:10]
            # max_workers = 1
        
        # Process sequentially if single worker or debug mode
        if max_workers <= 1:
            model = self.create_model(0, use_local_model, use_single_endpoint)
            
            print(f"Processing {len(entries)} questions sequentially")
            for entry in tqdm(entries, desc=f"Processing questions"):
                if self.cost_tracker.stop_requested:
                    print(f"\nCost threshold reached. Stopping execution.")
                    break
                
                # In sequential mode, we can always show verbose output if enabled
                result = self.process_entry(entry, model, verbose_worker=True, **kwargs)
                
                if result:
                    results.append(result)
                    if output_file:
                        append_result(result, output_file)
                    if self.track_cost and "cost" in result:
                        self.cost_tracker.update_cost(result["cost"])
        else:
            # Parallel processing
            pool_type = "process" if use_process_pool else "thread"
            endpoint_type = "single" if use_single_endpoint else "multiple"
            print(f"Processing {len(entries)} questions with {max_workers} workers using {pool_type} pool and {endpoint_type} endpoint(s)")
            
            if use_process_pool:
                # Process-based parallelism (better for timeouts)
                # Use try/finally instead of `with` so we can call shutdown(wait=False)
                # and avoid blocking forever when a worker is stuck.
                executor = ProcessPoolExecutor(max_workers=max_workers)
                try:
                    # Submit all tasks
                    futures = []
                    future_to_entry = {}
                    future_submit_time = {}

                    # Get the processor class module and name for dynamic import
                    processor_class_module = self.__class__.__module__
                    processor_class_name = self.__class__.__name__

                    print(f"Submitting {len(entries)} tasks to process pool...")
                    for i, entry in enumerate(entries):
                        worker_id = i % max_workers
                        future = executor.submit(
                            process_entry_in_process,
                            entry,
                            worker_id,
                            self.model_kwargs,
                            use_local_model,
                            worker_id == 0,  # Only first worker shows output
                            processor_class_module,
                            processor_class_name,
                            use_single_endpoint,
                            **{k: v for k, v in kwargs.items() if k != 'self'}
                        )
                        futures.append(future)
                        future_to_entry[future] = entry
                        future_submit_time[future] = time.time()
                    print(f"All {len(entries)} tasks submitted. Processing...")

                    # Track completed tasks
                    results = []
                    completed_tasks = 0

                    # Set up progress display
                    with tqdm(total=len(entries), desc="Processing questions") as pbar:
                        remaining_futures = set(futures)

                        while remaining_futures:
                            # --- per-task timeout: skip futures that have run too long ---
                            if per_task_timeout and per_task_timeout > 0:
                                now = time.time()
                                timed_out_futures = set()
                                for f in remaining_futures:
                                    elapsed = now - future_submit_time[f]
                                    if elapsed > per_task_timeout:
                                        entry = future_to_entry[f]
                                        q_snippet = str(entry.get("question", ""))[:80]
                                        print(f"\n[per_task_timeout] {elapsed:.0f}s elapsed "
                                              f"(limit {per_task_timeout}s), skipping: {q_snippet!r}")
                                        f.cancel()  # no-op if already running, that's fine
                                        timeout_result = {
                                            "question": entry.get("question", ""),
                                            "error": f"per_task_timeout after {elapsed:.0f}s",
                                            "log_data": None,
                                            "cost": 0.0,
                                        }
                                        results.append(timeout_result)
                                        if output_file:
                                            append_result(timeout_result, output_file)
                                        timed_out_futures.add(f)
                                        completed_tasks += 1
                                        pbar.update(1)
                                remaining_futures -= timed_out_futures

                            if not remaining_futures:
                                break

                            # --- collect any futures that finished in the last second ---
                            done_futures = set()
                            try:
                                for future in as_completed(remaining_futures, timeout=1.0):
                                    done_futures.add(future)
                                    try:
                                        result = future.result()
                                        if result:
                                            results.append(result)
                                            if output_file:
                                                append_result(result, output_file)
                                            if self.track_cost and "cost" in result:
                                                self.cost_tracker.update_cost(result["cost"])
                                    except Exception as e:
                                        print(f"Error processing entry: {e}")
                                    completed_tasks += 1
                                    pbar.update(1)
                            except TimeoutError:
                                # No futures completed within the 1-second poll — fine.
                                pass

                            remaining_futures -= done_futures

                            # Check if we should stop processing
                            if self.cost_tracker.stop_requested:
                                print(f"\nCost threshold reached. Stopping execution.")
                                for f in remaining_futures:
                                    f.cancel()
                                break

                    return results
                finally:
                    # Don't block on stuck worker processes.  With the HTTP-level
                    # timeout on VLLMServerModel (default 120 s) the workers will
                    # self-terminate shortly; but even if they don't, we exit cleanly.
                    executor.shutdown(wait=False, cancel_futures=True)
            else:
                # Thread-based parallelism (faster startup but less reliable timeouts)
                models = self.create_models(max_workers, use_local_model, use_single_endpoint)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Process function that selects the right model for each worker
                    def process_func(entry, model_idx):
                        return self.process_entry(
                            entry, 
                            models[model_idx],
                            verbose_worker=(model_idx == 0),  # Only first worker shows output
                            **kwargs
                        )
                    
                    # Submit all tasks
                    futures = []
                    for i, entry in enumerate(entries):
                        futures.append(executor.submit(process_func, entry, i % max_workers))
            
            # Common code for processing results from either executor
            for future in tqdm(as_completed(futures), total=len(entries), desc=f"Processing questions"):
                if self.cost_tracker.stop_requested:
                    print(f"\nCost threshold reached. Stopping execution.")
                    for f in futures:
                        f.cancel()
                    break
                    
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if output_file:
                            append_result(result, output_file)
                        if self.track_cost and "cost" in result:
                            self.cost_tracker.update_cost(result["cost"])
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue
        
        return results 
