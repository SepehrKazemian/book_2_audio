import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    pipeline, # Added pipeline here
)
from peft import prepare_model_for_kbit_training
from langchain.llms import HuggingFacePipeline
from typing import Tuple, Optional, Any

# --- Constants ---
DEFAULT_BASE_MODEL_ID = "stabilityai/stablelm-zephyr-3b"
# This specific pad token ID was hardcoded in the original model_caller.
# Verify if this is correct for the intended model/tokenizer.
DEFAULT_PAD_TOKEN_ID = 18610

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def quantization_config() -> BitsAndBytesConfig:
    """
    Creates and returns a BitsAndBytesConfig object configured for 4-bit quantization.

    Uses float16 compute dtype and double quantization.

    Returns:
        BitsAndBytesConfig: Configuration object for 4-bit quantization.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_quant_type="nf4", # Often default, commented out in original
        bnb_4bit_compute_dtype=torch.float16, # Use float16 for computation
        bnb_4bit_use_double_quant=True, # Use double quantization
    )
    logging.debug("Created 4-bit quantization config.")
    return quant_config


def unified_model_loader(
    model_id: str,
    use_gguf: bool = False,
    filename: Optional[str] = None,
    llama: bool = False, # Changed default to False based on model_caller usage
    trust_remote_code: bool = False,
    use_quantization: bool = True,
) -> Optional[Tuple[Any, Any]]:
    """
    Loads a Hugging Face model and tokenizer, with options for GGUF, Llama-specific loading,
    quantization, and trusting remote code.

    Args:
        model_id (str): The identifier or path for the Hugging Face model.
        use_gguf (bool, optional): If True, attempts to load using a GGUF file. Defaults to False.
        filename (str, optional): The name of the GGUF file if `use_gguf` is True. Defaults to None.
        llama (bool, optional): If True and not GGUF, loads using LlamaForCausalLM/LlamaTokenizer.
                                Otherwise, uses AutoModelForCausalLM/AutoTokenizer. Defaults to False.
        trust_remote_code (bool, optional): Trust remote code execution. Defaults to False.
        use_quantization (bool, optional): Apply 4-bit quantization if not using GGUF. Defaults to True.

    Returns:
        Optional[Tuple[Any, Any]]: A tuple (model, tokenizer), or None if loading fails.
    """
    model = None
    tokenizer = None
    quant_config = quantization_config() if use_quantization else None
    load_kwargs = {"device_map": "auto", "trust_remote_code": trust_remote_code}

    if use_quantization and not use_gguf:
         load_kwargs["quantization_config"] = quant_config
         logging.info(f"Attempting to load model {model_id} with 4-bit quantization.")
    else:
         logging.info(f"Attempting to load model {model_id} without explicit quantization.")

    try:
        if use_gguf and filename is not None:
            logging.info(f"Loading GGUF model: {filename}")
            # GGUF loading doesn't typically use BitsAndBytesConfig quantization_config arg
            if "quantization_config" in load_kwargs:
                 del load_kwargs["quantization_config"]
            model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename, use_fast=True, device_map="auto")
        else:
            if llama:
                logging.info("Loading using LlamaForCausalLM.")
                # Pass config directly for Llama if quantizing
                if use_quantization:
                     load_kwargs["config"] = quant_config
                model = LlamaForCausalLM.from_pretrained(model_id, **load_kwargs)
                # Use LlamaTokenizer specifically
                tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=True, device_map="auto", legacy=False)
                # Add pad token if needed (verify if still necessary for your Llama version)
                if tokenizer.pad_token is None:
                     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                     logging.info("Added special PAD token for Llama.")
            else:
                logging.info("Loading using AutoModelForCausalLM.")
                model = AutoModelForCausalLM.from_pretrained(model_id, revision="main", **load_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, device_map="auto")
                # Ensure pad token is set, defaulting to eos token if necessary
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logging.info("Set pad_token to eos_token.")

        logging.info(f"Successfully loaded model and tokenizer for {model_id}")
        return model, tokenizer

    except OSError as e:
        logging.error(f"OS Error loading model {model_id}: {e}. Check path/ID.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading model {model_id}: {e}", exc_info=True)
        return None


def model_caller(
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    use_quantization: bool = True,
    trust_remote_code: bool = True, # Defaulting based on original usage
    llama_model: bool = False, # Defaulting based on original usage
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID
) -> Optional[Tuple[Any, Any]]:
    """
    Loads a base model using unified_model_loader, prepares it for k-bit training,
    and configures the tokenizer for fine-tuning.

    Args:
        base_model_id (str, optional): Identifier for the base model. Defaults to DEFAULT_BASE_MODEL_ID.
        use_quantization (bool, optional): Whether to load the base model with quantization. Defaults to True.
        trust_remote_code (bool, optional): Trust remote code for base model loading. Defaults to True.
        llama_model (bool, optional): Is the base model a Llama model? Defaults to False.
        pad_token_id (int, optional): Specific ID to set for the pad token. Defaults to DEFAULT_PAD_TOKEN_ID.

    Returns:
        Optional[Tuple[Any, Any]]: A tuple (prepared_model, configured_tokenizer), or None if loading/preparation fails.
    """
    logging.info(f"Calling model loader for base model: {base_model_id}")
    loaded_data = unified_model_loader(
        base_model_id,
        llama=llama_model,
        trust_remote_code=trust_remote_code,
        use_quantization=use_quantization,
    )

    if loaded_data is None:
        logging.error("Failed to load base model and tokenizer in model_caller.")
        return None
    model, tokenizer = loaded_data

    try:
        logging.info("Preparing model for k-bit training...")
        # Prepare model for training (e.g., adds hooks, ensures correct dtype)
        model = prepare_model_for_kbit_training(model)
        logging.info("Model prepared for k-bit training.")

        # Configure model settings often used for fine-tuning
        model.config.use_cache = False # Disable caching for training efficiency
        model.config.pretraining_tp = 1 # Set tensor parallelism degree (1 means no parallelism)
        model.gradient_checkpointing_enable() # Enable gradient checkpointing to save memory
        logging.info("Enabled gradient checkpointing and configured model for training.")

        # Configure tokenizer settings
        tokenizer.padding_side = "right" # Set padding side (important for some models)
        tokenizer.add_eos_token = True # Ensure EOS token is added
        # Setting specific pad_token_id - ensure this is correct for your model/task
        tokenizer.pad_token_id = pad_token_id
        logging.info(f"Configured tokenizer: padding_side='right', add_eos_token=True, pad_token_id={pad_token_id}")

        return model, tokenizer

    except Exception as e:
        logging.error(f"Error preparing model or configuring tokenizer: {e}", exc_info=True)
        return None


def llm_pipeline(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 1024,
    top_k: int = 50,
    temperature: float = 0.01,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0, # Was 1.0 in this file's original
    dtype: torch.dtype = torch.bfloat16,
    do_sample: bool = True,
    truncation: bool = True,
    return_full_text: bool = False,
) -> Optional[HuggingFacePipeline]:
    """
    Creates a Hugging Face text generation pipeline wrapped in a LangChain HuggingFacePipeline.
    (Similar to the one in root llm_model_loader, but potentially with different defaults).

    Args:
        model: The loaded Hugging Face model (potentially PEFT-adapted).
        tokenizer: The loaded Hugging Face tokenizer.
        max_new_tokens (int, optional): Max tokens to generate. Defaults to 1024.
        top_k (int, optional): Top-k filtering value. Defaults to 50.
        temperature (float, optional): Sampling temperature. Defaults to 0.01.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
        repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.0.
        dtype (torch.dtype, optional): Data type for the pipeline. Defaults to torch.bfloat16.
        do_sample (bool, optional): Whether to use sampling. Defaults to True.
        truncation (bool, optional): Whether to truncate input. Defaults to True.
        return_full_text (bool, optional): Return full text including prompt. Defaults to False.

    Returns:
        Optional[HuggingFacePipeline]: A LangChain pipeline object, or None if creation fails.
    """
    try:
        logging.info("Creating text generation pipeline...")
        # Check bfloat16 support
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
             logging.warning("torch.bfloat16 requested but not supported by CUDA device. Pipeline might fail or use a different dtype.")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
            torch_dtype=dtype,
            repetition_penalty=repetition_penalty,
            device_map="auto",
            do_sample=do_sample,
            truncation=truncation,
            return_full_text=return_full_text,
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("Successfully created LangChain HuggingFacePipeline.")
        return local_llm

    except Exception as e:
        logging.error(f"Failed to create text generation pipeline: {e}", exc_info=True)
        return None
