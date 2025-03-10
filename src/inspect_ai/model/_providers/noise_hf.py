import asyncio
import copy
import functools
import json
import os
import gc
import tempfile
import time  # Added for timing
import sys
import random
import re
import warnings
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Thread
from typing import Any, Optional, Literal, Protocol, cast, List, Dict, Tuple, TypeVar
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    set_seed,
)
from typing_extensions import override
import psutil

# Add imports for PEFT and vLLM
try:
    from peft import LoraConfig, get_peft_model
    from vllm import LLM, SamplingParams, AsyncLLMEngine
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from inspect_ai._util.constants import DEFAULT_MAX_TOKENS
from inspect_ai._util.content import ContentText
from inspect_ai.tool import ToolChoice, ToolInfo

from .._chat_message import ChatMessage, ChatMessageAssistant
from .._generate_config import GenerateConfig
from .._model import ModelAPI
from .._model_output import (
    ChatCompletionChoice,
    Logprob,
    Logprobs,
    ModelOutput,
    ModelUsage,
    TopLogprob,
)
from .util import ChatAPIHandler, HFHandler

HF_TOKEN = "HF_TOKEN"


# Check if display is turned off via environment variable
DISABLE_TIMERS = False  # Always show timers regardless of INSPECT_DISPLAY setting


# Create a TimerManager to collect and report timing statistics
class TimerManager:
    """Manages timing data collection and reporting."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_time = None
        self.end_time = None
        self.gpu_metrics = {}
        
    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.start_time = None
        self.end_time = None
        
    def start_generation(self):
        """Mark the start of a generation sequence."""
        self.reset()
        self.start_time = time.perf_counter()
        
    def end_generation(self):
        """Mark the end of a generation sequence."""
        self.end_time = time.perf_counter()
        
    def add_timing(self, name: str, elapsed: float):
        """Add a timing measurement."""
        self.timings[name].append(elapsed)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timing data."""
        summary = {}
        
        # Calculate total time if we have start and end times
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            summary["total_generation_time"] = total_time
        
        # Process individual timing categories
        for name, times in self.timings.items():
            if times:
                category_summary = {
                    "total": sum(times),
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times)
                }
                
                # Calculate percentage of total time if available
                if "total_generation_time" in summary:
                    category_summary["percent"] = (category_summary["total"] / summary["total_generation_time"]) * 100
                
                summary[name] = category_summary
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of timing data."""
        # Always print summary regardless of DISABLE_TIMERS setting
        summary = self.get_summary()
        if not summary:
            print("\nNo timing data collected.")
            return
        
        # Extract total generation time if available
        total_time = summary.get("total_generation_time", 0)
        
        # Group timings by major categories
        major_categories = {
            "Initialization": ["vLLM model initialization", "Model loading"],
            "LoRA Operations": ["Get target modules", "LoRA config creation", "Get PEFT model", 
                               "Injecting noise into LoRA weights", "Saving LoRA adapter"],
            "Noise Injection": ["Adding noise to all weights", "Adding noise to percentage of weights", "Injecting noise"],
            "Generation Steps": ["Chat formatting", "Tokenization", "Generation", "LogProbs computation", "Decoding"],
            "HF Operations": ["HF model generation"],
            "vLLM Operations": ["vLLM generation"],
        }
        
        # Process each category
        for category_name, timing_keys in major_categories.items():
            category_data = {k: v for k, v in summary.items() if k in timing_keys and isinstance(v, dict)}
            
            if category_data:
                print(f"\n{category_name}:")
                print("-" * 70)
                
                # Sort by total time (descending)
                sorted_items = sorted(category_data.items(), key=lambda x: x[1]["total"], reverse=True)
                
                # Calculate category totals
                category_total = sum(item[1]["total"] for item in sorted_items)
                category_percent = (category_total / total_time * 100) if total_time else 0
                
                # Print each timing in this category
                for name, data in sorted_items:
                    percent = data.get("percent", 0)
                    
                # Print category summary
        
        # Show uncategorized timings
        all_categorized = [item for sublist in major_categories.values() for item in sublist]
        uncategorized = {k: v for k, v in summary.items() 
                         if k not in all_categorized and k != "total_generation_time" and isinstance(v, dict)}
        
        if uncategorized:
            print("\nOther Operations:")
            print("-" * 70)
            sorted_items = sorted(uncategorized.items(), key=lambda x: x[1]["total"], reverse=True)
            
            for name, data in sorted_items:
                percent = data.get("percent", 0)
                print(f"  {name:<30} {data['total']:.4f}s ({percent:.1f}% of total)")
        
        print("======================================================================")


# Create a global TimerManager instance
timer_manager = TimerManager()


# Add a modified Timer utility class that uses the TimerManager
class Timer:
    """Timer for measuring execution time that reports to TimerManager."""
    
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        elapsed_time = time.perf_counter() - self.start_time
        # Always add to timer manager regardless of display setting
        timer_manager.add_timing(self.name, elapsed_time)
        # Only conditionally print individual timings
        if not DISABLE_TIMERS:
            print(f"[TIMER] {self.name}: {elapsed_time:.4f} seconds")


# Create a GarbageCollector utility class for more intelligent memory management
class GarbageCollector:
    """Intelligent garbage collection to reduce overhead."""
    
    def __init__(self):
        self.gc_counter = 0
        self.last_full_gc_time = 0
        self.min_interval_seconds = 10  # Minimum time between full collections
        
    def collect(self, force=False):
        """Collect garbage selectively based on thresholds and timers.
        
        Args:
            force (bool): Force garbage collection regardless of thresholds
        """
        current_time = time.time()
        self.gc_counter += 1
        
        # Always clear CUDA cache when requested
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Check if we should do a full garbage collection
        should_collect = force
        
        # Time-based throttling - don't collect too frequently
        if current_time - self.last_full_gc_time >= self.min_interval_seconds:
            # After every 5 potential GC points, do a full collection
            if self.gc_counter >= 5:
                should_collect = True
            
            # Get memory info if available
            try:
                import psutil
                mem = psutil.virtual_memory()
                # If memory usage is above 85%, force collection
                if mem.percent > 85:
                    should_collect = True
            except ImportError:
                # If psutil isn't available, use a simpler strategy
                should_collect = self.gc_counter >= 3
        
        # Perform actual garbage collection if needed
        if should_collect:
            if not DISABLE_TIMERS:
                print(f"[GC] Performing full garbage collection (counter: {self.gc_counter})")
            gc.collect()
            self.gc_counter = 0
            self.last_full_gc_time = current_time
            return True
            
        return False


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""

    mean: float
    std: float
    percentage: float
    # LoRA specific parameters
    use_lora: bool  # No default value, must be explicitly set
    is_noisy: bool = False
    seed: Optional[int] = None
    lora_r: int = 8  # Updated default to a valid vLLM rank
    lora_adapter_path: Optional[str] = None  # Path to save/load LoRA adapter
    target_modules: List[str] = field(default_factory=list)  # Use default_factory for mutable default

    def __post_init__(self):
        # Validate configuration
        if self.use_lora:
            # Validate LoRA rank
            valid_ranks = [8, 16, 32, 64, 128, 256]
            if self.lora_r not in valid_ranks:
                raise ValueError(f"LoRA rank must be one of {valid_ranks}, got {self.lora_r}")
        
        # General validation
        if not (0.0 <= self.percentage <= 1.0):
            raise ValueError(f"noise_percentage must be between 0.0 and 1.0, got {self.percentage}")
        if self.std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {self.std}")


class NoiseHuggingFaceAPI(ModelAPI):
    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        base_url: str = None,
        api_key: str = None,
        std: float = 0.0,
        seed: int = 42,
        config: GenerateConfig = GenerateConfig(),
        **model_args,
    ):
        """Initialize the noise provider.

        Args:
            model_name: Name of the model (from parent ModelAPI).
            model_path: Path to the model.
            base_url: Alternate base URL for model.
            api_key: API key for model.
            std: Standard deviation for the noise.
            seed: Random seed.
            config: Model configuration.
            **model_args: Additional model arguments.
        """
        # Initialize garbage collector
        self.gc_manager = GarbageCollector()
        
        # First handle model_name from parent class, use it as model_path if no model_path provided
        if model_name is not None and model_path is None:
            model_path = model_name

        # Ensure we have a model_path
        if model_path is None:
            raise ValueError("Either model_name or model_path must be provided")

        # Store model_path to use for loading the model
        self.model_path = model_path
        
        # Get tokenizer path or default to model path
        self.tokenizer_path = model_args.pop("tokenizer_path", model_path)
        
        # Simplify: Just use the constructor parameter directly
        self.seed = seed
        print(f"Using seed value: {self.seed} (from constructor)")
        
        # Initialize parent class with only the parameters it accepts
        super().__init__(
            model_name=model_path,
            base_url=base_url,
            api_key=api_key,
            config=config
        )

        # Set up model and device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # If API key wasn't passed, try to get it from environment variable
        if self.api_key is None:
            self.api_key = os.environ.get(HF_TOKEN)
            if self.api_key is None:
                print("Warning: HF_TOKEN environment variable not set. Some models may not be accessible.")
        
        # Get noise parameters
        noise_std = model_args.pop("noise_std", std)  # Use noise_std if provided, otherwise use std
        
        # Extract LoRA parameters
        use_lora = model_args.pop("use_lora", None)
        # Handle both string and boolean values for use_lora
        if isinstance(use_lora, str):
            self.use_lora = use_lora.lower() == "true"
        elif use_lora is None:
            raise ValueError("use_lora parameter is required. Set it to True or False explicitly.")
        else:
            self.use_lora = bool(use_lora)
            
        lora_r = int(model_args.pop("lora_r", 8))
        
        # If using LoRA, validate rank against vLLM supported values
        if self.use_lora:
            valid_ranks = [8, 16, 32, 64, 128, 256]
            if lora_r not in valid_ranks:
                raise ValueError(
                    f"When use_lora=True, lora_r must be one of {valid_ranks}. Got {lora_r}."
                )
        
        # Handle target modules
        lora_target_modules = model_args.pop("lora_target_modules", None)
        if lora_target_modules == "auto":
            lora_target_modules = None  # Will be auto-detected
        
        # Initialize noise configuration
        self.noise_config = NoiseConfig(
            mean=model_args.pop("noise_mean", 0.0),
            percentage=model_args.pop("noise_percentage", 1.0),
            std=noise_std,
            seed=self.seed,  # Use the constructor seed
            use_lora=self.use_lora,
            lora_r=lora_r,
            target_modules=lora_target_modules,
        )
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.vllm_model = None
        self.temp_dir = None  # Add this line to initialize temp_dir
        
        # Load the model and tokenizer
        self.load_model_and_tokenizer()

        # Add initialization of _is_closed
        self._is_closed = False

        # Remove storage of original weights
        self.original_weights = None
        
        # Store model loading args for reloading
        self.model_args = model_args

        # Set random seeds - Simplified
        try:
            set_random_seeds(self.seed)
        except Exception as e:
            print(f"WARNING: Error setting random seeds with value {self.seed}: {str(e)}")

        # Collect known model_args (then delete them so we can pass the rest on)
        def collect_model_arg(name: str) -> Any | None:
            nonlocal model_args
            value = model_args.get(name, None)
            if value:
                model_args.pop(name)
            return value

        device = collect_model_arg("device")
        tokenizer = collect_model_arg("tokenizer")
        model_path = collect_model_arg("model_path")  # Collect model_path first
        if model_path:  # Only assign if not None
            self.model_path = model_path  # Then assign it to self
        tokenizer_path = collect_model_arg("tokenizer_path")
        if tokenizer_path:  # Only assign if not None
            self.tokenizer_path = tokenizer_path
        self.batch_size = collect_model_arg("batch_size")
        self.chat_template = collect_model_arg("chat_template")
        self.tokenizer_call_args = collect_model_arg("tokenizer_call_args")
        if self.tokenizer_call_args is None:
            self.tokenizer_call_args = {}

        # Device configuration
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # Model loading - KEEP ON CPU INITIALLY
        model_kwargs = {
            "device_map": "cpu",  # Load to CPU first
            "low_cpu_mem_usage": True,
            **model_args,
        }
        
        # Only add auth token if it's available
        if self.api_key is not None:
            model_kwargs["use_auth_token"] = self.api_key
            
        if self.model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
        else:
            # If model_path is None, raise a more descriptive error
            raise ValueError("Model path is not defined. Please provide a valid model_path.")

        # Keep model on CPU to save GPU memory, we'll move to GPU only when needed
        # self.model = self.model.to(self.device)  # Don't move to GPU yet

        # Add cleanup on deletion
        self._is_closed = False

        # Tokenizer loading
        if tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif self.model_path:
            if self.tokenizer_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            raise ValueError("Neither model path nor tokenizer path is defined. Please provide at least one of them.")

        # LLMs generally don't have a pad token and we need one for batching
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if not (0.0 <= self.noise_config.percentage <= 1.0):
            raise ValueError("noise_percentage must be between 0.0 and 1.0")
        if self.noise_config.std < 0.0:
            raise ValueError("noise_std must be non-negative")

    def __del__(self):
        """Clean up resources when the object is deleted."""
        self.close()

    def close(self):
        """Close the model and clean up resources."""
        if self._is_closed:
            return

        # Clean up LoRA resources if using LoRA
        if self.noise_config.use_lora:
            # Mark as closed before cleanup to ensure proper resource management
            self._is_closed = True
            self.cleanup_lora()
        else:
            # For non-LoRA usage, we can release the HF model
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None
                # Force a collection here as we're closing
                self.gc_manager.collect(force=True)
            
            self._is_closed = True

    def reset_weights(self):
        """Reload model from HF cache instead of storing weights."""
        if self._is_closed:
            return

        # Delete current model
        del self.model
        # Use selective garbage collection
        self.gc_manager.collect()

        # Reload model from cache
        if self.model_path:
            model_kwargs = {
                "device_map": "cpu",
                "low_cpu_mem_usage": True,
                **self.model_args,
            }
            
            # Only add auth token if it's available
            if self.api_key is not None:
                model_kwargs["use_auth_token"] = self.api_key
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            # Keep model on CPU to save GPU memory, we'll move to GPU only when needed
            # self.model = self.model.to(self.device)  # Don't move to GPU yet
        else:
            raise ValueError("Model path not set, cannot reset weights")

        # Reset noise flag
        self.noise_config.is_noisy = False

    def get_target_modules(self):
        """Get target modules for LoRA adaptation.
        
        If custom target modules are specified in the config, use those.
        Otherwise, detect all linear layers in the model.
        """
        # Check if custom target modules are specified
        if hasattr(self.noise_config, 'target_modules') and self.noise_config.target_modules:
            print(f"Using custom target modules: {self.noise_config.target_modules}")
            return self.noise_config.target_modules
        
        # Detect all linear layers
        import re
        model_modules = str(self.model.modules)
        pattern = r'\((\w+)\): Linear'
        linear_layer_names = re.findall(pattern, model_modules)

        names = []
        for name in linear_layer_names:
            names.append(name)
        
        target_modules = list(set(names))
        print(f"Detected {len(target_modules)} linear layers for LoRA adaptation")
        return target_modules

    def get_target_modules_properly(self):
        """
        Determine the target modules for LoRA based on model architecture.
        Automatically detects appropriate linear layers for the model.
        
        Returns:
            list: List of module names to apply LoRA to
        """
        if self.noise_config.target_modules:
            # If explicitly specified, use those
            print(f"Using explicitly specified target modules: {self.noise_config.target_modules}")
            return self.noise_config.target_modules
            
        # Default to all linear layers based on model architecture
        # This is a common pattern for transformer models
        
        # Find all linear layer names in the model
        target_modules = []
        excluded_modules = ["lm_head", "head", "output"]
        
        # Track module sizes for statistics
        all_linear_layers = {}
        excluded_layers = {}
        included_layers = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Extract the general module type (last part of the name)
                parts = name.split('.')
                if len(parts) > 0:
                    module_type = parts[-1]
                    shape = (module.weight.shape[0], module.weight.shape[1])  # (output_dim, input_dim)
                    all_linear_layers[name] = {
                        "type": module_type,
                        "shape": shape,
                        "params": module.weight.numel()
                    }
                    
                    # Skip modules that are known to cause issues with vLLM
                    if any(excluded in name for excluded in excluded_modules):
                        print(f"EXCLUDED: {name} - shape {shape}")
                        excluded_layers[name] = all_linear_layers[name]
                        continue
                    
                    included_layers[name] = all_linear_layers[name]
                        
                    if module_type not in target_modules:
                        target_modules.append(module_type)
        
        # Compute statistics
        total_params = sum(layer["params"] for layer in all_linear_layers.values())
        included_params = sum(layer["params"] for layer in included_layers.values())
        excluded_params = sum(layer["params"] for layer in excluded_layers.values())
        
        # If no modules found, raise a clear error
        if not target_modules:
            raise ValueError("No linear layers were detected in the model. Cannot auto-detect target modules for LoRA. Go to get_target_modules() to manually specify target modules.")
            
        return target_modules

    def create_noise_lora_adapter(self):
        """Create a LoRA adapter with random noise using PEFT."""
        from peft import LoraConfig, get_peft_model
        
        # Set seed if configured
        if self.noise_config.seed is not None:
            self.set_seed(self.noise_config.seed)
            print(f"Using seed {self.noise_config.seed} for noise generation")
        
        # Create a persistent directory to store the adapter
        adapter_name = f"noise_adapter_{self.noise_config.seed}"
        adapter_dir = os.path.join("./lora_adapters", adapter_name)
        os.makedirs(adapter_dir, exist_ok=True)
        
        with Timer("Get target modules"):
            # Define LoRA configuration
            target_modules = self.get_target_modules_properly()  # Improved function
        
        with Timer("LoRA config creation"):
            lora_config = LoraConfig(
                r=self.noise_config.lora_r,
                lora_alpha=self.noise_config.lora_r,  # Usually set to same as r
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
        
        # Apply LoRA config to the model
        with Timer("Get PEFT model"):
            peft_model = get_peft_model(self.model, lora_config)
        
        # Collect layer statistics before injecting noise
        lora_a_shapes = {}
        lora_b_shapes = {}
        module_types = {}
        
        # Inject noise directly into the LoRA weights
        with Timer("Injecting noise into LoRA weights"):
            with torch.no_grad():
                for name, param in peft_model.named_parameters():
                    # Only modify LoRA adapter weights (lora_A and lora_B)
                    if 'lora_A' in name or 'lora_B' in name:
                        # Track statistics about layers
                        module_name = name.split('.')[0]
                        module_types[module_name] = module_name
                        
                        if 'lora_A' in name:
                            lora_a_shapes[name] = param.shape
                        elif 'lora_B' in name:
                            lora_b_shapes[name] = param.shape
                        
                        # Generate noise
                        noise = torch.normal(
                            mean=self.noise_config.mean,
                            std=self.noise_config.std,
                            size=param.shape,
                            device=param.device,
                            dtype=param.dtype
                        )
                        # Apply noise
                        param.add_(noise)
        
        # Save the adapter
        with Timer("Saving LoRA adapter"):
            peft_model.save_pretrained(adapter_dir)
        
        # Store the adapter path
        self.noise_config.lora_adapter_path = adapter_dir
        self.noise_config.is_noisy = True
        
        # Clean up
        del peft_model
        # Use selective garbage collection
        self.gc_manager.collect()
        
        return adapter_dir
    
    def initialize_vllm(self):
        """Initialize vLLM model for LoRA-based inference."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for LoRA-based noise generation. Please install vLLM and PEFT packages.")
        
        # Check if LoRA is explicitly enabled
        if not self.noise_config.use_lora:
            raise ValueError("LoRA must be explicitly enabled for vLLM initialization. Set use_lora=True in the model configuration.")
        
        # Validate LoRA rank for vLLM compatibility
        valid_ranks = [8, 16, 32, 64, 128, 256]
        if self.noise_config.lora_r not in valid_ranks:
            raise ValueError(f"vLLM only supports LoRA ranks of {valid_ranks}. Current value: {self.noise_config.lora_r}")
        
        if self.vllm_model is None:
            print(f"Initializing vLLM model with {self.model_path}")
            print(f"Using tokenizer from {self.tokenizer_path}")
            
            try:
                # Set environment variable for HF token if available
                if self.api_key is not None:
                    os.environ["HF_TOKEN"] = self.api_key
                
                with Timer("vLLM model initialization"):
                    # Initialize vLLM model with enable_lora=True to support multi-LoRA
                    self.vllm_model = LLM(
                        model=self.model_path,
                        tokenizer=self.tokenizer_path,
                        tensor_parallel_size=1,
                        max_lora_rank=self.noise_config.lora_r,
                        trust_remote_code=True,
                        enable_lora=True,  # Enable LoRA support
                        seed=42
                    )
            except Exception as e:
                error_msg = f"Error initializing vLLM model: {e}"
                print(error_msg)
                raise RuntimeError(error_msg) from e
    
    def cleanup_lora(self):
        """Clean up LoRA resources."""
        print("Cleaning up LoRA resources")
        
        # Clean up vLLM model
        if self.vllm_model is not None:
            # We don't want to delete the vLLM model as it contains the base model
            # which we want to keep loaded on the GPU
            print("vLLM model will be retained for future use")
            
            # If we have a specific adapter loaded, we can attempt to unload it
            # but vLLM doesn't have a direct way to unload specific adapters yet
            pass
        
        # Clean up temporary directory if we're completely done
        if self.temp_dir is not None and self._is_closed:
            try:
                self.temp_dir.cleanup()
                self.temp_dir = None
                print("Temporary directory cleaned up")
            except Exception as e:
                print(f"Error cleaning up temporary directory: {str(e)}")
            
        # Reset adapter path (but don't clear is_noisy flag to avoid regenerating)
        if self._is_closed:
            self.noise_config.lora_adapter_path = None
            self.noise_config.is_noisy = False
        
        # Use smart garbage collection - force it when cleaning up
        self.gc_manager.collect(force=True)

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set seed for random distributions."""
        if seed is not None:
            torch.manual_seed(seed)  # seed for CPU operations
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)  # seed for GPU operations
            np.random.seed(seed)  # seed for numpy operations

    @torch.inference_mode()
    def add_noise_all(
        self, batch_size: int = None
    ):  # batch_size param kept for API compatibility
        """Add noise to all weights in the model layer by layer."""
        try:
            # Set seed if configured
            if self.noise_config.seed is not None:
                self.set_seed(self.noise_config.seed)
                print(f"Using seed {self.noise_config.seed} for noise generation")

            with Timer("Adding noise to all weights"):
                for layer_i, (name, param) in enumerate(self.model.named_parameters()):
                    # Generate noise for entire layer at once
                    noise = torch.normal(
                        mean=self.noise_config.mean,
                        std=self.noise_config.std,
                        size=param.shape,
                        device=self.device,
                        dtype=param.dtype,
                    )

                    # Add noise to the entire layer
                    param.add_(noise)

                    # Clean up layer memory
                    del noise

            self.noise_config.is_noisy = True

        except Exception as e:
            print(f"Error in noise injection: {str(e)}")
            self.reset_weights()  # Reset weights if there's an error
            raise
        finally:
            # Use smart GC management
            self.gc_manager.collect()

    @torch.inference_mode()
    def add_noise_percentage(self, batch_size: int = 20 * 10**6):
        """Add noise to a percentage of weights."""
        # Handle full noise case separately for efficiency
        if self.noise_config.percentage == 1.0:
            self.add_noise_all(batch_size=batch_size)
            return

        try:
            # Set seed if configured
            if self.noise_config.seed is not None:
                self.set_seed(self.noise_config.seed)
                print(f"Using seed {self.noise_config.seed} for noise generation")

            with Timer("Adding noise to percentage of weights"):
                for name, param in self.model.named_parameters():
                    param_size = param.numel()

                    # Process in batches for memory efficiency
                    for start in range(0, param_size, batch_size):
                        end = min(start + batch_size, param_size)
                        current_batch_size = end - start
                        n_batch_noise = int(
                            current_batch_size * self.noise_config.percentage
                        )

                        if n_batch_noise == 0:
                            continue

                        # Sample indices for this batch
                        indices = torch.randint(
                            start, end, (n_batch_noise,), device=self.device
                        ).unique()

                        # Generate noise
                        noise = torch.normal(
                            mean=self.noise_config.mean,
                            std=self.noise_config.std,
                            size=(len(indices),),
                            device=self.device,
                            dtype=param.dtype,
                        )

                        param.view(-1)[indices] += noise
                        del noise, indices  # Just delete tensors

            # Use smart garbage collection
            self.gc_manager.collect()
            self.noise_config.is_noisy = True

        except Exception as e:
            print(f"Error in noise injection: {str(e)}")
            self.reset_weights()
            raise

    def inject_noise(self):
        """Main method to inject noise based on configuration."""
        if not self.noise_config.std:
            return

        try:
            if not self.noise_config.is_noisy:
                if self.noise_config.use_lora:
                    # Use LoRA-based noise generation
                    if not VLLM_AVAILABLE:
                        raise ImportError("vLLM and PEFT are required for LoRA-based noise generation")
                    
                    # Initialize vLLM model first if not already initialized
                    if self.vllm_model is None:
                        self.initialize_vllm()
                    
                    # Create noise LoRA adapter
                    adapter_path = self.create_noise_lora_adapter()
                    
                    # Mark as noisy
                    self.noise_config.is_noisy = True
                    
                    print(f"Created noise LoRA adapter at {adapter_path}")
                else:
                    # Use traditional noise injection
                    if self.noise_config.percentage == 1.0:
                        self.add_noise_all()
                    else:
                        self.add_noise_percentage()
        except Exception as e:
            print(f"Error in noise injection: {str(e)}")
            if self.noise_config.use_lora:
                self.cleanup_lora()
            else:
                self.reset_weights()

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response with noise injection.

        Args:
            input: List of chat messages.
            tools: List of tools.
            tool_choice: Tool choice.
            config: Generate configuration.

        Returns:
            Model output.
        """
        # Start timing the entire generation process
        timer_manager.start_generation()
        
        try:
            # Use vLLM for LoRA-based generation
            if self.use_lora:
                # Ensure vLLM model is initialized
                if self.vllm_model is None:
                    self.initialize_vllm()
                
                # Create noise LoRA adapter if needed
                if not self.noise_config.is_noisy and (self.noise_config.std > 0):
                    self.inject_noise()
                
                try:
                    # Use vLLM with LoRA for generation
                    result = await self._generate_vllm(input, tools, tool_choice, config)
                    return result
                except asyncio.CancelledError as e:
                    print("vLLM generation was cancelled, attempting fallback")
                    # Create a fallback response to avoid breaking the evaluation
                    return self._create_fallback_response(input, tools, "Generation was cancelled")
                except Exception as e:
                    print(f"Error during vLLM generation: {type(e).__name__}: {str(e)}")
                    # Create a fallback response with error details
                    return self._create_fallback_response(input, tools, f"Error during generation: {str(e)}")
            else:
                # For traditional noise injection, use HF model
                # Add noise if configured
                if not self.noise_config.is_noisy and (self.noise_config.std > 0):
                    self.inject_noise()
                
                # We do NOT need to move model to GPU and back since device_map="auto" 
                # already handles this efficiently
                result = await self._generate_hf(input, tools, tool_choice, config)
                return result
        except Exception as e:
            print(f"Unexpected error in generate: {type(e).__name__}: {str(e)}")
            # Return fallback response if all else fails
            return self._create_fallback_response(input, tools, "An unexpected error occurred during generation")
        finally:
            # End timing for the entire generation process
            timer_manager.end_generation()
            # Print the timing summary
            # timer_manager.print_summary()

    def _create_fallback_response(self, input: list[ChatMessage], tools: list[ToolInfo], error_message: str) -> ModelOutput:
        """Create a fallback response when generation fails.
        
        Args:
            input: List of chat messages.
            tools: List of tools.
            error_message: Error message to include in response.
            
        Returns:
            Fallback model output.
        """
        fallback_text = f"I apologize, but I was unable to generate a proper response. {error_message}."
        
        return ModelOutput(
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessageAssistant(content=fallback_text, source="generate"),
                    finish_reason="stop",
                )
            ],
            model=self.model_name,
            usage=ModelUsage(
                input_tokens=0,  # We don't have accurate token counts for fallback
                completion_tokens=len(fallback_text.split()),  # Rough estimate
                total_tokens=len(fallback_text.split()),  # Rough estimate
            ),
        )

    @override
    def max_tokens(self) -> int | None:
        """Default is 16, bump it up to a value suitable for evals."""
        return DEFAULT_MAX_TOKENS

    @override
    def max_connections(self) -> int:
        """Effectively the batch size."""
        return 1

    @override
    def collapse_user_messages(self) -> bool:
        return True

    def hf_chat(self, messages: list[ChatMessage], tools: list[ToolInfo]) -> str:
        # convert to hf format
        tools_list = []
        hf_messages = copy.deepcopy(messages)
        
        # Convert ChatMessage objects to dictionaries for HF format
        hf_messages_dict = []
        for msg in hf_messages:
            msg_dict = {"role": msg.role}
            if isinstance(msg.content, str):
                msg_dict["content"] = msg.content
            elif hasattr(msg, "content") and msg.content is not None:
                # Handle content that might be a list or other structure
                if isinstance(msg.content, list):
                    # For content that's a list (like with image inputs)
                    content_text = ""
                    for item in msg.content:
                        if hasattr(item, "text"):
                            content_text += item.text
                    msg_dict["content"] = content_text
                else:
                    # For other types of content
                    msg_dict["content"] = str(msg.content)
            hf_messages_dict.append(msg_dict)
        
        # Use the tokenizer_apply_chat function
        chat = tokenizer_apply_chat(
            self.tokenizer,
            hf_messages_dict,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # return
        return cast(str, chat)

    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.model_path}")
        print(f"Loading tokenizer from {self.tokenizer_path}")
        
        # Load tokenizer from the specified path
        tokenizer_kwargs = {
            "use_fast": True,
            "padding_side": "left",
        }
        
        # Only add auth token if it's available
        if self.api_key is not None:
            tokenizer_kwargs["use_auth_token"] = self.api_key
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            **tokenizer_kwargs
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on noise injection method
        if not self.use_lora:
            # For traditional noise injection, load model directly
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",  # Use auto device mapping for efficient memory usage
            }
            
            # Only add auth token if it's available
            if self.api_key is not None:
                model_kwargs["use_auth_token"] = self.api_key
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
        else:
            # For LoRA, initialize vLLM model with proper configurations
            self.initialize_vllm()

    def generate_noise(self, parameter, noise_config=None):
        """Generate noise based on the noise configuration.

        Args:
            parameter: Parameter to apply noise to.
            noise_config: Optional noise configuration to override the default.

        Returns:
            Parameter with noise applied.
        """
        # Use local noise config if provided, otherwise use default
        config = noise_config or self.noise_config
        
        # Use PyTorch's normal distribution to generate noise
        if config.is_noisy and np.random.random() < config.percentage:
            # Fix the random seed if specified
            if config.seed is not None:
                torch.manual_seed(config.seed)
                np.random.seed(config.seed)
            
            # Generate noise according to specified distribution
            noise = torch.randn_like(parameter) * config.std + config.mean
            return parameter + noise
        else:
            return parameter

    async def _generate_vllm(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        try:
            # Add noise if configured
            if not self.noise_config.is_noisy and (self.noise_config.std > 0):
                with Timer("Injecting noise"):
                    self.inject_noise()

            # Create handler
            handler: ChatAPIHandler | None = (
                HFHandler(self.model_name) if len(tools) > 0 else None
            )

            # Create chat
            with Timer("Chat formatting"):
                chat = self.hf_chat(input, tools)
            
            # Use vLLM for inference with LoRA adapter
            if self.noise_config.use_lora and self.noise_config.is_noisy:
                if not VLLM_AVAILABLE:
                    raise ImportError("vLLM and PEFT are required for LoRA-based inference. Please ensure both are installed.")
                
                # Ensure vLLM model is initialized
                if self.vllm_model is None:
                    with Timer("vLLM initialization"):
                        self.initialize_vllm()
                
                # Verify the adapter path exists
                if not self.noise_config.lora_adapter_path or not os.path.exists(self.noise_config.lora_adapter_path):
                    raise ValueError(f"LoRA adapter path does not exist or is not set: {self.noise_config.lora_adapter_path}")
                
                # Create a unique adapter ID based on seed for vLLM to track
                adapter_id = f"noise_{self.noise_config.seed}"
                
                # Create LoRA request
                lora_request = LoRARequest(
                    adapter_id, 
                    1,  # Adapter ID
                    self.noise_config.lora_adapter_path
                )
                
                # Set up sampling parameters
                sampling_params = SamplingParams(
                    temperature=config.temperature if config.temperature is not None else 0.0,
                    top_p=config.top_p if config.top_p is not None else 1.0,
                    max_tokens=config.max_tokens or DEFAULT_MAX_TOKENS,
                )
                
                print(f"Generating with vLLM using LoRA adapter {adapter_id} at {self.noise_config.lora_adapter_path}")
                
                # Print GPU stats before generation
                # print_gpu_stats("GPU stats before vLLM generation")
                
                try:
                    # Use asyncio timeout context manager to prevent hanging
                    timeout_seconds = 120  # 2 minutes timeout
                    
                    # Generate with vLLM, protected by timeout
                    async def generate_with_timeout():
                        # vLLM generate is not async, so run it in executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        with Timer("vLLM generation"):
                            result = await loop.run_in_executor(
                                None, 
                                lambda: self.vllm_model.generate(
                                    chat,
                                    sampling_params,
                                    lora_request=lora_request
                                )
                            )
                        return result
                    
                    # Use asyncio.wait_for for the timeout
                    outputs = await asyncio.wait_for(
                        generate_with_timeout(),
                        timeout=timeout_seconds
                    )
                    
                    # Print GPU stats after generation
                    # print_gpu_stats("GPU stats after vLLM generation")
                    
                    # Check if outputs is valid
                    if not outputs or len(outputs) == 0 or not hasattr(outputs[0], 'outputs') or len(outputs[0].outputs) == 0:
                        print("vLLM returned empty or invalid outputs")
                        raise RuntimeError("vLLM returned empty or invalid outputs")
                    
                    # Extract generated text
                    generated_text = outputs[0].outputs[0].text
                    
                    if not generated_text or len(generated_text.strip()) == 0:
                        print("vLLM generated empty text")
                        raise RuntimeError("vLLM generated empty text")
                    # Create a wrapper object with output attribute for consistency
                    wrapped_output = GenerateVLLMOutput(output=generated_text)
                    
                    # Get token counts if available
                    input_tokens = getattr(outputs[0], "prompt_token_ids", [])
                    output_tokens = getattr(outputs[0].outputs[0], "token_ids", [])
                    
                    # Create model output
                    return ModelOutput(
                        choices=[
                            ChatCompletionChoice(
                                index=0,
                                message=chat_completion_assistant_message(
                                    wrapped_output, tools, handler, self.model_name
                                ),
                                finish_reason="stop",
                            )
                        ],
                        model=self.model_name,
                        usage=ModelUsage(
                            input_tokens=len(input_tokens) if input_tokens else 0,
                            completion_tokens=len(output_tokens) if output_tokens else 0,
                            total_tokens=(len(input_tokens) if input_tokens else 0) + 
                                         (len(output_tokens) if output_tokens else 0),
                        ),
                    )
                except Exception as e:
                    print(f"Error during vLLM generation: {str(e)}")
                    # We still want to raise the error for proper handling
                    raise
            else:
                # This should not be reached with the current implementation
                raise ValueError("vLLM model initialized but LoRA not enabled - this is an unexpected state")

        except asyncio.CancelledError:
            print("vLLM generation task was cancelled")
            # Clean up any resources if needed
            raise  # Re-raise the cancellation
        except Exception as e:
            print(f"Unexpected error in vLLM generation: {type(e).__name__}: {str(e)}")
            raise
        # finally:
            # Final GPU stats
            # print_gpu_stats("Final GPU stats after vLLM generation")

    async def _generate_hf(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        try:
            # Add noise if configured
            if not self.noise_config.is_noisy and (self.noise_config.std > 0):
                with Timer("Injecting noise"):
                    self.inject_noise()

            # Create handler
            handler: ChatAPIHandler | None = (
                HFHandler(self.model_name) if len(tools) > 0 else None
            )

            # Create chat
            with Timer("Chat formatting"):
                chat = self.hf_chat(input, tools)
            
            # Use standard HF inference
            assert isinstance(self.tokenizer_call_args, dict)
            # Prepare tokenizer
            tokenizer = functools.partial(
                self.tokenizer,
                return_tensors="pt",
                padding=True,
                **self.tokenizer_call_args,
            )

            # Prepare generator
            kwargs: dict[str, Any] = dict(do_sample=True)
            if config.max_tokens is not None:
                kwargs["max_new_tokens"] = config.max_tokens
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                kwargs["top_k"] = config.top_k
            if config.logprobs is not None:
                kwargs["output_logits"] = config.logprobs
            if "return_dict_in_generate" in kwargs:
                assert kwargs["return_dict_in_generate"]
            kwargs["return_dict_in_generate"] = True
            
            # Note: No need to explicitly move model to device since device_map="auto" handles this
            generator = functools.partial(self.model.generate, **kwargs)

            # Prepare decoder
            decoder = functools.partial(
                self.tokenizer.batch_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Generate (uses a queue to batch so we await)
            with Timer("HF model generation"):
                response = await batched_generate(
                    GenerateInput(
                        input=chat,
                        device=self.model.device,
                        tokenizer=tokenizer,
                        generator=generator,
                        decoder=decoder,
                        batch_size=config.max_connections or self.max_connections(),
                    )
                )

            # Gather logprobs
            final_logprobs = None
            if config.logprobs is not None:
                with Timer("Extracting logprobs"):
                    final_logprobs = extract_logprobs(
                        response=response,
                        top=config.top_logprobs,
                        tokenizer=self.tokenizer,
                    )

            # Construct choice
            choice = ChatCompletionChoice(
                message=chat_completion_assistant_message(
                    response.output, tools, handler, self.model_name
                ),
                logprobs=(
                    Logprobs(content=final_logprobs)
                    if final_logprobs is not None
                    else None
                ),
            )

            # Return output
            return ModelOutput(
                model=self.model_name,
                choices=[choice],
                usage=ModelUsage(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                ),
            )
        finally:
            # We don't move the model to CPU since we're using device_map="auto"
            # Just clear CUDA cache selectively - don't call gc.collect() here
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def shorten_tool_id(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Shorten the tool_call_id in the messages to the last 9 characters for Mistral."""
    for i, message in enumerate(messages):
        if message.role == "tool":
            # Trim tool_call_id in tool messages
            if message.tool_call_id is not None:
                message.tool_call_id = message.tool_call_id[-9:]
        elif message.role == "assistant" and hasattr(message, "tool_calls"):
            # Trim tool_call IDs inside tool_calls for assistant messages
            for tool_call in message.tool_calls or []:
                tool_call.id = tool_call.id[-9:]
    return messages


def tools_to_mistral_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tools to the format required for Mistral."""
    mistral_tools = []
    for tool in tools:
        mistral_tools.append(
            {
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": tool["parameters"]["type"],
                        "properties": tool["parameters"]["properties"],
                        "required": tool["parameters"]["required"],
                    },
                }
            }
        )
    return mistral_tools


def inspect_tools_to_string(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Convert tools to a string for Qwen."""
    for message in messages:
        if message.role == "assistant":
            # check if the message contains a tool call
            tool_content = ""
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_content += f'\n```json\n{{"name": "{tool_call.function}", "arguments": {json.dumps(tool_call.arguments)}}}\n```'
            # remove the tool call from the message
            message.tool_calls = None
            if isinstance(message.content, str):
                message.content += tool_content
            else:
                message.content.append(ContentText(text=tool_content))
    return messages


def chat_completion_assistant_message(
    response: Any,
    tools: list[ToolInfo],
    handler: ChatAPIHandler | None,
    model_name: str,
) -> ChatMessageAssistant:
    if handler:
        # If it's a string, we need to wrap it
        if isinstance(response, str):
            return handler.parse_assistant_response(response, tools)
        # Otherwise it should have an output attribute
        return handler.parse_assistant_response(response.output, tools)
    else:
        # If it's a string, use it directly
        if isinstance(response, str):
            return ChatMessageAssistant(content=response, source="generate")
        # Otherwise extract the output attribute
        return ChatMessageAssistant(content=response.output, source="generate")


def set_random_seeds(seed: int | None = None) -> None:
    if seed is None:
        seed = np.random.default_rng().integers(2**32 - 1)
    # python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # transformers seed
    set_seed(seed)


# return value from generate as a result of specifying return_dict_in_generate
class ModelGenerateOutput:
    sequences: Tensor
    logits: tuple[Tensor]


class Tokenizer(Protocol):
    def __call__(
        self, input: list[str]
    ) -> dict[Literal["input_ids", "attention_mask"], Tensor]: ...


class Generator(Protocol):
    def __call__(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: ...


class Decoder(Protocol):
    def __call__(self, sequences: Tensor) -> list[str]: ...


@dataclass
class GenerateInput:
    input: str
    device: str
    tokenizer: Tokenizer
    generator: Generator
    decoder: Decoder
    batch_size: int


@dataclass
class GenerateOutput:
    output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    logprobs: torch.Tensor | None


@dataclass
class _QueueItem:
    input: GenerateInput
    future: asyncio.Future[GenerateOutput]
    loop: asyncio.AbstractEventLoop


batch_thread: Thread | None = None

batch_queue: "Queue[_QueueItem]" = Queue()


async def batched_generate(input: GenerateInput) -> GenerateOutput:
    # start the background thread if necessary
    global batch_thread
    if batch_thread is None:
        batch_thread = Thread(target=process_batches, daemon=True)
        batch_thread.start()

    # enqueue the job
    loop = asyncio.get_event_loop()
    future: asyncio.Future[GenerateOutput] = loop.create_future()
    batch_queue.put(_QueueItem(input=input, future=future, loop=loop))

    # await the job
    await future

    # return it
    return future.result()


def process_batches() -> None:
    while True:
        # drain the queue (wait until no new messages have shown up for 2 seconds)
        inputs: list[tuple[GenerateInput, asyncio.Future[GenerateOutput]]] = []
        while True:
            try:
                input = batch_queue.get(timeout=2)
                loop = input.loop
                inputs.append((input.input, input.future))
                if len(inputs) == input.input.batch_size:
                    # max batch size reached
                    break
            except Empty:
                # we have exhausted the queue
                break

        # see if we have any work to do
        if len(inputs) == 0:
            continue

        try:
            # capture the generator and decoder functions
            first_input = inputs[0][0]
            device = first_input.device
            tokenizer = first_input.tokenizer
            generator = first_input.generator
            decoder = first_input.decoder

            # tokenize and move to device
            with Timer("Tokenization"):
                tokenized_inputs = tokenizer([item[0].input for item in inputs])
                input_ids = tokenized_inputs["input_ids"]
                attention_mask = tokenized_inputs["attention_mask"]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

            # generate
            with torch.inference_mode():
                with Timer("Generation"):
                    generation_outputs = cast(
                        ModelGenerateOutput,
                        generator(input_ids=input_ids, attention_mask=attention_mask),
                    )
                    generate_ids = generation_outputs.sequences
                    logits = generation_outputs.logits

            # get logprobs from logits
            logprobs = None
            if logits is not None:
                with Timer("LogProbs computation"):
                    stacked_logits = torch.stack(logits).transpose(0, 1)
                    logprobs = torch.nn.functional.log_softmax(stacked_logits, dim=-1)

            # decode
            with Timer("Decoding"):
                generated_tokens = generate_ids[:, input_ids.size(dim=1) :]
                if logprobs is not None:
                    assert logprobs.shape[1] == generated_tokens.shape[1]
                outputs = decoder(sequences=generated_tokens)

            # call back futures
            for i, output in enumerate(outputs):
                future = inputs[i][1]
                input_tokens = input_ids.size(dim=1)
                output_tokens = generate_ids.size(dim=1) - input_ids.size(dim=1)

                # asyncio futures are not thread safe, so we need to pass the event loop
                # down to this point, so we can mark the future as done in a thread safe manner.
                # see: https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading
                loop.call_soon_threadsafe(
                    future.set_result,
                    GenerateOutput(
                        output=output,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        logprobs=logprobs[i] if logprobs is not None else None,
                    ),
                )

        except Exception as ex:
            for inp in inputs:
                future = inp[1]
                loop.call_soon_threadsafe(future.set_exception, ex)


def extract_logprobs(
    response: GenerateOutput,
    top: int | None,
    tokenizer: PreTrainedTokenizerBase,
) -> list[Logprob]:
    assert response.logprobs is not None
    k = top or 1
    topk_values, topk_inds = response.logprobs.topk(k=k, dim=-1)
    final_logprobs = []
    for toks, vals in zip(topk_inds, topk_values):
        top_logprobs: list[TopLogprob] = []
        for tok, val in zip(toks, vals):
            # TODO: you get byte artifacts converting single ids to tokens like this...
            # but `tokenizer.decode` strips spaces. There must be a better way to do this.
            token_str = tokenizer.convert_ids_to_tokens(tok.item())
            top_logprobs.append(
                TopLogprob(
                    token=token_str,
                    logprob=val,
                    bytes=list(map(ord, token_str)),
                )
            )
        final_logprobs.append(
            Logprob(
                token=top_logprobs[0].token,
                logprob=top_logprobs[0].logprob,
                bytes=top_logprobs[0].bytes,
                top_logprobs=top_logprobs,
            )
        )
    return final_logprobs


def convert_chat_style_prompt_to_str(messages, add_generation_prompt: bool = False) -> str:
    """Convert a list of messages to a string. Adds the a last 'Assistant:' if add_generation_prompt is True."""

    items = []
    for p in messages:
        if p["role"] == "user":
            items.append("User: " + p["content"])
        elif p["role"] == "assistant":
            items.append("Assistant: " + p["content"])
        elif p["role"] == "system":
            items.append("System: " + p["content"])
        else:
            raise ValueError(f"Unknown role: {p['role']}")

    out = "\n\n".join(items)
    if add_generation_prompt:
        if len(out) > 0:
            out = out + "\n\n"
        out = out + "Assistant:"

    return out


def tokenizer_apply_chat(tokenizer, messages, tokenize=True, add_generation_prompt=False, **kwargs):
    """Apply the tokenizer to a list of messages."""

    if tokenizer.chat_template is None:
        out_s = convert_chat_style_prompt_to_str(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return tokenizer.encode(out_s, **kwargs)
        else:
            assert len(kwargs) == 0
            return out_s
    else:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
        )


@dataclass
class GenerateVLLMOutput:
    """Wrapper class for vLLM generation output to provide consistent interface."""
    output: str


# Simple GPU monitoring utility - using only PyTorch built-in functions
def get_gpu_stats():
    """Get current GPU memory usage statistics using only PyTorch built-in functions."""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    stats = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # MB
        reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
        allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
        free_memory = total_memory - reserved_memory
        
        stats.append({
            "device": i,
            "total_memory_mb": round(total_memory, 1),
            "reserved_memory_mb": round(reserved_memory, 1),
            "allocated_memory_mb": round(allocated_memory, 1),
            "free_memory_mb": round(free_memory, 1),
        })
    
    return stats

def print_gpu_stats(label="Current GPU stats"):
    """Print current GPU memory usage with a label."""
    stats = get_gpu_stats()
    if isinstance(stats, str):
        print(f"{label}: {stats}")
        return
    
    print(f"\n{label}:")
    print("-" * 80)
    for gpu in stats:
        print(f"GPU {gpu['device']}:")
        print(f"  Memory: {gpu['allocated_memory_mb']:.1f}MB / {gpu['total_memory_mb']:.1f}MB allocated")
        print(f"  Reserved: {gpu['reserved_memory_mb']:.1f}MB (Free: {gpu['free_memory_mb']:.1f}MB)")
    print("-" * 80)
