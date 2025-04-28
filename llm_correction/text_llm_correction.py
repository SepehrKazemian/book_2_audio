import pickle
import logging
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional, Generator
from peft import PeftModel

# --- Setup Logging (Initialize early) ---
# Use basicConfig with force=True to ensure it's set even if run multiple times or by other modules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- Standard Imports (assuming script run from project root or root added to PYTHONPATH) ---
try:
    from llm_finetuning.training import model_loader as ml
    from text_cleaning import text_cleaning as tc
    # Use absolute import for modules in the same package (assuming run from root)
    from llm_correction import result_check as rc
    logging.debug("Successfully imported modules.")
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}. Ensure the script is run from the project root or the root is in PYTHONPATH.")
    raise # Re-raise the error to stop execution if imports fail

# --- Constants ---
# !! IMPORTANT: Model path should be configurable or relative to project structure !!
# Assuming model saved by fine-tuning step is used:
DEFAULT_MODEL_PATH = "llm_finetuning/training/fine_tuned_model"
DEFAULT_NUM_MODELS = 1 # Number of model instances to load (for potential parallelism)
DEFAULT_USE_QUANTIZATION = True
DEFAULT_MAX_TOKENS_PER_BATCH = 1024
# Index in the cleaned lines list where the main content starts (adjust if needed)
INTRODUCTION_IDX = 153
# File to save intermediate and final results (relative to project root)
# !! IMPORTANT: Consider making paths configurable !!
OUTPUT_PICKLE_FILE = "data/text_correction_results.pkl"
# How often to save intermediate results (number of batches)
SAVE_INTERVAL = 20
# Prompt template for the LLM correction task
LLM_PROMPT_TEMPLATE = (
    "<|system|>\nCorrect the spelling mistakes and find the book sections in the "
    "following sentences. Do not change content or do not add any extra sentences."
    "<|endoftext|>\n<|user|>\n{batch_text}<|endoftext|>\n<|assistant|>\n"
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_multiple_models(
    model_path: str, num_models: int, use_quantization: bool = True
) -> Optional[Tuple[List[Any], Any, Any]]:
    """
    Loads multiple instances of a PEFT model, potentially prepared for inference.

    Args:
        model_path (str): Path to the saved PEFT adapter.
        num_models (int): Number of model instances to load.
        use_quantization (bool): Whether the base model was loaded with quantization.

    Returns:
        Optional[Tuple[List[Any], Any, Any]]: A tuple containing:
            - List of loaded model pipeline objects (from ml.llm_pipeline).
            - The tokenizer object (from the last loaded model).
            - The base model object (from the last loaded model).
            Returns None if loading fails.
    """
    models = []
    last_tokenizer = None
    last_base_model = None
    logging.info(f"Attempting to load {num_models} model instances from path: {model_path}")

    for i in range(num_models):
        logging.info(f"Loading model instance {i + 1}/{num_models}...")
        try:
            # Load base model and tokenizer using the fine_tuning model_loader
            # model_caller prepares for k-bit training, might not be ideal for pure inference
            # Consider adding an inference-specific loading function to model_loader
            base_model, tokenizer = ml.model_caller(use_quantization=use_quantization)
            if base_model is None or tokenizer is None:
                 logging.error(f"Failed to load base model/tokenizer for instance {i+1}.")
                 continue # Try next instance or return None?

            # Load the PEFT adapter
            peft_model = PeftModel.from_pretrained(base_model, model_path)
            logging.info(f"Loaded PEFT adapter for instance {i+1}.")

            # Create an inference pipeline
            # Ensure ml.llm_pipeline uses appropriate settings for inference
            model_pipeline = ml.llm_pipeline(peft_model, tokenizer)
            if model_pipeline is None:
                 logging.error(f"Failed to create LLM pipeline for instance {i+1}.")
                 continue

            models.append(model_pipeline)
            last_tokenizer = tokenizer
            last_base_model = peft_model # Store the PEFT model as 'bare_model' reference
            logging.info(f"Successfully loaded and created pipeline for model instance {i + 1}.")

        except Exception as e:
            logging.error(f"Error loading model instance {i + 1}: {e}", exc_info=True)
            # Decide whether to continue or fail entirely
            # return None # Example: Fail if any instance fails

    if not models:
        logging.error("Failed to load any model instances.")
        return None

    # Return list of pipelines, the last tokenizer, and the last PEFT model instance
    return models, last_tokenizer, last_base_model


def batch_tokenizer(
    text_list: List[str], tokenizer: Any, max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH
) -> Generator[str, None, None]:
    """
    Generates batches of text from a list, ensuring each batch does not exceed max_tokens.

    Args:
        text_list (List[str]): List of text strings to batch.
        tokenizer (Any): The tokenizer object with encode/decode methods.
        max_tokens (int): Maximum number of tokens allowed per batch.

    Yields:
        str: A string representing a batch of text.
    """
    current_batch_tokens: List[int] = []
    current_token_count: int = 0
    logging.info(f"Starting batch tokenization. Max tokens per batch: {max_tokens}")

    for text in text_list:
        if not text:
             continue # Skip empty lines
        try:
            # Add newline separation between texts in a batch
            # Encode with add_special_tokens=False as we handle structure manually
            tokens = tokenizer.encode(text + "\n", add_special_tokens=False)
            token_len = len(tokens)

            # If adding this text exceeds the max token limit, yield the current batch
            if current_token_count + token_len > max_tokens and current_batch_tokens:
                yield tokenizer.decode(current_batch_tokens, skip_special_tokens=True) # Decode the batch
                logging.debug(f"Yielded batch with {current_token_count} tokens.")
                current_batch_tokens = tokens # Start the next batch with the current tokens
                current_token_count = token_len
                # Handle case where a single text exceeds max_tokens
                if token_len > max_tokens:
                     logging.warning(f"Single text item exceeds max_tokens ({token_len} > {max_tokens}). Truncating or skipping might be needed. Yielding as is for now.")
                     yield tokenizer.decode(current_batch_tokens, skip_special_tokens=True)
                     current_batch_tokens = []
                     current_token_count = 0

            else:
                # Otherwise, add the tokens to the current batch
                current_batch_tokens.extend(tokens)
                current_token_count += token_len

        except Exception as e:
            logging.warning(f"Could not encode text: '{text[:50]}...'. Skipping. Error: {e}")
            continue

    # Yield the final batch if there's any remaining tokens
    if current_batch_tokens:
        yield tokenizer.decode(current_batch_tokens, skip_special_tokens=True)
        logging.debug(f"Yielded final batch with {current_token_count} tokens.")


def process_batch(batch_text: str, model_pipeline: Any) -> Optional[str]:
    """
    Processes a single batch of text using the provided LLM pipeline.

    Args:
        batch_text (str): The text batch to process.
        model_pipeline (Any): The loaded LLM pipeline object (e.g., from LangChain).

    Returns:
        Optional[str]: The processed text result from the model, or None on error.
    """
    try:
        prompt = LLM_PROMPT_TEMPLATE.format(batch_text=batch_text)
        # Assuming the pipeline handles max_new_tokens etc. internally based on its setup
        # Or pass parameters here if needed: result = model_pipeline(prompt, max_new_tokens=...)
        result = model_pipeline(prompt)
        # Result might be a list or dict depending on pipeline, extract text
        if isinstance(result, list) and result:
             # Example: [{'generated_text': '...'}]
             if isinstance(result[0], dict) and 'generated_text' in result[0]:
                  return result[0]['generated_text']
        elif isinstance(result, str): # Direct string output
             return result
        logging.warning(f"Unexpected model output format: {type(result)}. Full output: {result}")
        return str(result) # Fallback to string conversion

    except Exception as e:
        logging.error(f"Error processing batch: {batch_text[:100]}... Error: {e}", exc_info=True)
        return None


def process_batch_and_compare(
    batch_text: str, model_pipeline: Any, tokenizer: Any, bare_model: Any, batch_idx: int
) -> Optional[Dict[str, Any]]:
    """
    Helper function for parallel processing: processes a batch, compares the result,
    and returns structured data.

    Args:
        batch_text (str): The input text batch.
        model_pipeline (Any): The specific LLM pipeline instance to use.
        tokenizer (Any): The tokenizer.
        bare_model (Any): The base model (for result_check's similarity).
        batch_idx (int): The index of the batch for logging.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing original batch, processed result,
                                  and comparison scores, or None on failure.
    """
    logging.debug(f"Worker processing batch {batch_idx}...")
    processed_result = process_batch(batch_text, model_pipeline)

    if processed_result is None:
        logging.warning(f"Processing failed for batch {batch_idx}. Skipping comparison.")
        # Return data indicating failure or skip?
        return {
            "batch_index": batch_idx,
            "original_batch": batch_text,
            "processed_result": None,
            "comparison": None,
            "status": "processing_failed"
        }

    # Compare results using the refactored function from result_check
    comparison_scores = rc.compare_text_results(
        original_text=batch_text, # Compare against the original batch input
        processed_result=processed_result,
        tokenizer=tokenizer,
        model=bare_model # Pass the base model for embeddings
    )

    logging.debug(f"Finished comparison for batch {batch_idx}.")
    return {
        "batch_index": batch_idx,
        "original_batch": batch_text,
        "processed_result": processed_result,
        "comparison": comparison_scores,
        "status": "success"
    }


def parallel_batch_processing(
    batches: List[str],
    models: List[Any],
    tokenizer: Any,
    bare_model: Any,
    output_filename: str = OUTPUT_PICKLE_FILE,
    save_interval: int = SAVE_INTERVAL
) -> List[Optional[Dict[str, Any]]]:
    """
    Processes text batches in parallel using multiple model instances and saves results.

    Args:
        batches (List[str]): List of text batches to process.
        models (List[Any]): List of loaded LLM pipeline instances.
        tokenizer (Any): The tokenizer object.
        bare_model (Any): The base model object (for result comparison).
        output_filename (str): Path to save intermediate and final results.
        save_interval (int): How often (in batches) to save intermediate results.

    Returns:
        List[Optional[Dict[str, Any]]]: List containing results for each batch
                                        (or None if processing failed for that batch).
    """
    num_batches = len(batches)
    num_models = len(models)
    if num_models == 0:
        logging.error("No models provided for parallel processing.")
        return [None] * num_batches

    final_results: List[Optional[Dict[str, Any]]] = [None] * num_batches
    processed_count = 0

    logging.info(f"Starting parallel processing for {num_batches} batches using {num_models} model instances.")

    # Use ThreadPoolExecutor for potentially I/O bound tasks like model inference
    # Adjust max_workers based on resources and model behavior (GIL release etc.)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_models) as executor:
        future_to_batch_idx: Dict[concurrent.futures.Future, int] = {}

        # Submit tasks, cycling through available models
        for idx, batch_text in enumerate(batches):
            model_instance = models[idx % num_models] # Simple round-robin assignment
            future = executor.submit(
                process_batch_and_compare,
                batch_text,
                model_instance,
                tokenizer,
                bare_model,
                idx
            )
            future_to_batch_idx[future] = idx

        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_batch_idx):
            batch_idx = future_to_batch_idx[future]
            try:
                result_data = future.result()
                final_results[batch_idx] = result_data
                processed_count += 1
                status = result_data.get("status", "unknown") if result_data else "future_error"
                logging.info(f"Completed batch {batch_idx + 1}/{num_batches}. Status: {status}")

                # Periodically save the results
                if processed_count % save_interval == 0 and processed_count > 0:
                    logging.info(f"Saving intermediate results ({processed_count}/{num_batches} processed)...")
                    try:
                        with open(output_filename, "wb") as file:
                            pickle.dump(final_results, file)
                        logging.info(f"Intermediate results saved to {output_filename}")
                    except (IOError, pickle.PicklingError) as e:
                        logging.error(f"Failed to save intermediate results: {e}")

            except Exception as e:
                logging.error(f"Error retrieving result for batch {batch_idx}: {e}", exc_info=True)
                # Optionally store error information in final_results
                final_results[batch_idx] = {
                     "batch_index": batch_idx, "status": "future_error", "error": str(e)
                }

    logging.info(f"Finished parallel processing. Processed {processed_count}/{num_batches} batches.")
    # Final save after loop completion
    logging.info(f"Saving final results to {output_filename}...")
    try:
        with open(output_filename, "wb") as file:
            pickle.dump(final_results, file)
        logging.info("Final results saved successfully.")
    except (IOError, pickle.PicklingError) as e:
        logging.error(f"Failed to save final results: {e}")

    return final_results


if __name__ == "__main__":
    logging.info("--- Starting LLM Text Correction Script ---")

    # --- 1. Load Models ---
    loaded_models, tokenizer_obj, bare_model_obj = load_multiple_models(
        model_path=DEFAULT_MODEL_PATH,
        num_models=DEFAULT_NUM_MODELS,
        use_quantization=DEFAULT_USE_QUANTIZATION,
    )
    if not loaded_models or not tokenizer_obj or not bare_model_obj:
        logging.critical("Failed to load models. Exiting.")
        exit(1)
    logging.info(f"Successfully loaded {len(loaded_models)} model instance(s).")

    # --- 2. Load and Clean Text ---
    logging.info("Loading and cleaning text using text_cleaning module...")
    cleaned_lines = tc.main() # Assumes tc.main() returns List[str] or None
    if cleaned_lines is None:
        logging.critical("Text cleaning failed or returned no lines. Exiting.")
        exit(1)
    logging.info(f"Text cleaning complete. {len(cleaned_lines)} lines obtained.")

    # --- 3. Batch Text ---
    logging.info("Batching text...")
    # Use only lines after the introduction index
    lines_to_process = cleaned_lines[INTRODUCTION_IDX:]
    if not lines_to_process:
         logging.warning(f"No lines found after INTRODUCTION_IDX ({INTRODUCTION_IDX}). Processing might yield empty results.")

    batches_generator = batch_tokenizer(
        lines_to_process, tokenizer_obj, max_tokens=DEFAULT_MAX_TOKENS_PER_BATCH
    )
    # Convert generator to list for parallel processing function
    batches_list = list(batches_generator)
    if not batches_list:
         logging.critical("Batch tokenization resulted in zero batches. Exiting.")
         exit(1)
    logging.info(f"Created {len(batches_list)} batches.")

    # --- 4. Process Batches in Parallel ---
    # Note: The original script created a Dataset here, which isn't strictly necessary
    # if parallel_batch_processing accepts a list of strings.
    # data = {"text": batches_list}
    # dataset = Dataset.from_dict(data)
    # final_results_list = parallel_batch_processing(
    #     dataset["text"], loaded_models, tokenizer_obj, bare_model_obj
    # )
    final_results_list = parallel_batch_processing(
        batches_list, loaded_models, tokenizer_obj, bare_model_obj, output_filename=OUTPUT_PICKLE_FILE, save_interval=SAVE_INTERVAL
    )

    # --- 5. Post-processing/Analysis (Optional) ---
    # Add any analysis of final_results_list here if needed
    successful_batches = sum(1 for res in final_results_list if res and res.get("status") == "success")
    failed_batches = len(final_results_list) - successful_batches
    logging.info(f"Processing complete. Successful batches: {successful_batches}, Failed/Skipped: {failed_batches}")

    logging.info("--- LLM Text Correction Script Finished ---")
