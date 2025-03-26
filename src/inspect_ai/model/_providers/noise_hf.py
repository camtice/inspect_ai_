import asyncio
import copy
import json
import os
import gc
import tempfile
import time
import sys
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Literal, cast, List, Dict, Tuple, TypeVar
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from typing_extensions import override
import psutil

# Add imports for PEFT and vLLM
try:
    from peft import LoraConfig, get_peft_model
    from vllm import LLM, SamplingParams
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
    lora_r: int
    is_noisy: bool = False
    seed: Optional[int] = None
    lora_adapter_path: Optional[str] = None  # Path to save/load LoRA adapter
    target_modules: List[str] = field(default_factory=list)  # Use default_factory for mutable default

    def __post_init__(self):
        # Validate std (currently done in NoiseHuggingFaceAPI constructor)
        if self.std is None:
            raise ValueError("noise_std parameter is required. Set it to a float value explicitly.")
        if self.std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {self.std}")
            
        # Validate percentage
        if not (0.0 <= self.percentage <= 1.0):
            raise ValueError(f"noise_percentage must be between 0.0 and 1.0, got {self.percentage}")
        
        # Validate LoRA rank
        valid_ranks = [8, 16, 32, 64, 128, 256]
        if self.lora_r not in valid_ranks:
            raise ValueError(f"LoRA rank must be one of {valid_ranks}, got {self.lora_r}")
            
        # Validate seed
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {type(self.seed)}")
            
        # Validate lora_adapter_path
        if self.lora_adapter_path is not None and not isinstance(self.lora_adapter_path, str):
            raise ValueError(f"lora_adapter_path must be a string, got {type(self.lora_adapter_path)}")
            
        # Validate target_modules
        if not isinstance(self.target_modules, list):
            raise ValueError(f"target_modules must be a list, got {type(self.target_modules)}")


class NoiseHuggingFaceAPI(ModelAPI):
    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        base_url: str = None,
        api_key: str = None,
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
        
        # Store model path and seed
        self.model_path = model_path
        self.seed = seed

        # Process and extract model arguments
        self.device = model_args.pop("device", None)
        self.tokenizer_path = model_args.pop("tokenizer_path", None) or model_path
        self.batch_size = model_args.pop("batch_size", None)
        self.chat_template = model_args.pop("chat_template", None)
        self.max_model_len = model_args.pop("max_model_len", None)
        self.tokenizer_call_args = model_args.pop("tokenizer_call_args", {})

        # Default model kwargs
        self.base_model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            **model_args,  # Add remaining kwargs
        }

        # Initialize parent class
        super().__init__(
            model_name=model_path,
            base_url=base_url,
            api_key=api_key,
            config=config
        )

        # Try to get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get(HF_TOKEN)
            if self.api_key is None:
                print("Warning: HF_TOKEN environment variable not set. Some models may not be accessible.")

        # Get noise parameters and create config object
        noise_config = NoiseConfig(
            mean=model_args.pop("noise_mean", 0.0),
            std=model_args.pop("noise_std"),
            percentage=model_args.pop("noise_percentage", 1.0),
            seed=self.seed,
            lora_r=int(model_args.pop("lora_r", 8)),
            target_modules=model_args.pop("lora_target_modules", None) or [],
        )
        self.noise_config = noise_config

        # Set up device configuration
        if not self.device:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"

        # Add API token to base kwargs if available
        if self.api_key is not None:
            self.base_model_kwargs["use_auth_token"] = self.api_key

        # Initialize model and tokenizer to None
        self.model = None
        self.tokenizer = None
        self.vllm_model = None
        self.temp_dir = None
        
        # Load the model and tokenizer
        # self.load_model_and_tokenizer()
        self.load_tokenizer()
        self.initialize_vllm()

        # Setup is complete
        self._is_closed = False

    def __del__(self):
        """Clean up resources when the object is deleted."""
        self.close()

    def close(self):
        """Close the model and clean up resources."""
        if self._is_closed:
            return

        # Mark as closed before cleanup to ensure proper resource management
        self._is_closed = True
        self.cleanup_lora()

    def get_target_modules_properly(self, temp_model):
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
        # Find all linear layer names in the model
        target_modules = []
        excluded_modules = ["lm_head", "head", "output"]
        
        # Track module sizes for statistics
        all_linear_layers = {}
        excluded_layers = {}
        included_layers = {}
        
        for name, module in temp_model.named_modules():
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
        temp_model = None
        try:
            # Temporarily load the model
            temp_model = self.load_temp_model_for_lora()
            
            # Create adapter directory
            adapter_name = f"noise_adapter_{self.noise_config.seed}"
            adapter_dir = os.path.join("./lora_adapters", adapter_name)
            os.makedirs(adapter_dir, exist_ok=True)
            
            # Create LoRA config and inject noise
            lora_config = LoraConfig(
                r=self.noise_config.lora_r,
                lora_alpha=self.noise_config.lora_r,
                target_modules=self.get_target_modules_properly(temp_model),
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            peft_model = get_peft_model(temp_model, lora_config)

            self.set_seed(self.noise_config.seed)
            
            # Inject noise into LoRA weights
            with torch.no_grad():
                for name, param in peft_model.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name:
                        noise = torch.normal(
                            mean=self.noise_config.mean,
                            std=self.noise_config.std,
                            size=param.shape,
                            device=param.device,
                            dtype=param.dtype,
                        )
                        param.add_(noise)
            
            # Save the adapter
            peft_model.save_pretrained(adapter_dir)
            
            return adapter_dir
            
        finally:
            # Clean up temporary model
            if temp_model is not None:
                del temp_model
            self.gc_manager.collect(force=True)
    
    def initialize_vllm(self):
        """Initialize vLLM model for LoRA-based inference."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for LoRA-based noise generation. Please install vLLM and PEFT packages.")
        
        if self.vllm_model is None:
            print(f"Initializing vLLM model with {self.model_path}")
            print(f"Using tokenizer from {self.tokenizer_path}")
            
            try:
                # Set environment variable for HF token if available
                if self.api_key is not None:
                    os.environ["HF_TOKEN"] = self.api_key
                
                # Add warning for large context windows
                if self.max_model_len and self.max_model_len > 10000:
                    print(f"WARNING: Using max_model_len={self.max_model_len} which is >10k tokens. vLLM may have issues with very large context windows depending on your GPU memory and model size.")
                
                print(f"max_model_len: {self.max_model_len}")

                # Initialize vLLM model with enable_lora=True to support multi-LoRA
                self.vllm_model = LLM(
                    model=self.model_path,
                    tokenizer=self.tokenizer_path,
                    tensor_parallel_size=1,
                    max_lora_rank=self.noise_config.lora_r,
                    trust_remote_code=True,
                    enable_lora=True,  # Enable LoRA support
                    seed=42,
                    max_model_len=self.max_model_len
                )
            except Exception as e:
                error_msg = f"Error initializing model in initialize_vllm() vLLM model: {e}"
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

    def inject_noise(self):
        """Create a LoRA adapter with noise."""
        if not self.noise_config.std:
            return

        try:
            if not self.noise_config.is_noisy:
                # Check vLLM availability
                if not VLLM_AVAILABLE:
                    raise ImportError("vLLM and PEFT are required for LoRA-based noise generation")
                
                # Create noise LoRA adapter
                adapter_path = self.create_noise_lora_adapter()
                
                # Save the adapter path in the noise config
                self.noise_config.lora_adapter_path = adapter_path
                
                # Mark as noisy
                self.noise_config.is_noisy = True
                
                print(f"Created noise LoRA adapter at {adapter_path}")
        except Exception as e:
            print(f"Error in noise injection: {str(e)}")
            self.cleanup_lora()

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # Ensure vLLM model is initialized
        if self.vllm_model is None:
            self.initialize_vllm()
        
        # Create noise LoRA adapter if needed
        if not self.noise_config.is_noisy and (self.noise_config.std > 0):
            self.inject_noise()
        
        # Use vLLM with LoRA for generation
        return await self._generate_vllm(input, tools, tool_choice, config)

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
        return cast(str, chat)

    def get_model_kwargs(self, device_map: str = "auto") -> dict:
        """Get model initialization kwargs based on context.
        Returns:
            Dictionary of model initialization arguments
        """
        kwargs = self.base_model_kwargs.copy()
        kwargs["device_map"] = device_map
        return kwargs


    def load_tokenizer(self):
        """Load only the tokenizer."""
        tokenizer_kwargs = {
            "use_fast": True,
            "padding_side": "left",
        }
        
        if self.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, **tokenizer_kwargs)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def load_temp_model_for_lora(self):
        """Temporarily load the model for LoRA adapter creation."""
        print("Temporarily loading model for LoRA adapter creation")
        
        # Use same configuration approach as in load_model_and_tokenizer
        temp_kwargs = {
            "device_map": "cpu",  # Ensure loading on CPU
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
            
        # Now load the model with proper configuration
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **temp_kwargs
            )
            print(f"Successfully loaded temporary model from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading temporary model: {str(e)}")
            raise

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
                self.inject_noise()

            # Create handler
            handler: ChatAPIHandler | None = (
                HFHandler(self.model_name) if len(tools) > 0 else None
            )

            # Create chat
            chat = self.hf_chat(input, tools)
            
            # Use vLLM for inference with LoRA adapter
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM and PEFT are required for LoRA-based inference. Please ensure both are installed.")
            
            # Ensure vLLM model is initialized
            if self.vllm_model is None:
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
                temperature=config.temperature,
                top_p=config.top_p if config.top_p is not None else 1.0,
                max_tokens=config.max_tokens or DEFAULT_MAX_TOKENS,
            )
                                
            try:
                # Use asyncio timeout context manager to prevent hanging
                timeout_seconds = 120  # 2 minutes timeout
                
                # Generate with vLLM, protected by timeout
                async def generate_with_timeout():
                    # vLLM generate is not async, so run it in executor to avoid blocking
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, 
                        lambda: self.vllm_model.generate(
                            chat,
                            sampling_params,
                            lora_request=lora_request,
                            use_tqdm=False
                        )
                    )
                    return result
                
                # Use asyncio.wait_for for the timeout
                outputs = await asyncio.wait_for(
                    generate_with_timeout(),
                    timeout=timeout_seconds
                )
                
                # Check if outputs is valid
                if not outputs or len(outputs) == 0 or not hasattr(outputs[0], 'outputs') or len(outputs[0].outputs) == 0:
                    print("vLLM returned empty or invalid outputs")
                    raise RuntimeError("vLLM returned empty or invalid outputs")
                
                # Extract generated text
                generated_text = outputs[0].outputs[0].text
                
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

        except asyncio.CancelledError:
            print("vLLM generation task was cancelled")
            # Clean up any resources if needed
            raise  # Re-raise the cancellation
        except Exception as e:
            print(f"Unexpected error in vLLM generation: {type(e).__name__}: {str(e)}")
            raise


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
