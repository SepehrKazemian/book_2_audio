import multiprocessing as mp
import pickle
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional

# Use absolute imports from the project root directory
try:
    from llm_finetuning.data_preparation import dataset_manipulation as ds_man
    from llm_finetuning.data_preparation import text_to_sample as text2sample
    from llm_finetuning.data_preparation import dataset_to_text as ds2txt
    from llm_finetuning.data_preparation import noise_generator as noiseGen
except ImportError as e:
    logging.error(f"Failed to import data preparation submodules: {e}. Ensure script is run from project root or root is in PYTHONPATH.")
    # Define dummy functions or exit if critical
    raise

# --- Constants ---
DATASET_NAME = "google/wiki40b"
DATASET_CONFIG = "en"
TOKENIZER_ID = "stabilityai/stablelm-zephyr-3b"
DEFAULT_SPLITS = ["train", "validation", "test"]
DEFAULT_NOISE_PROB = 0.1
DEFAULT_NUM_PROCESSES = 12 # Adjust based on system capabilities
DEFAULT_TOTAL_SAMPLES = 1_000_000
DEFAULT_MAX_TOKENS_PER_SAMPLE = 490
DEFAULT_CHUNK_SIZE = 100_000
OUTPUT_FILENAME_TEMPLATE = "wiki_{split}_data.pkl"
PROMPT_TEXT = "Correct the spelling mistakes and find the book sections in the following sentences without changing content or add any more sentences."

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_process_datasets(
    dataset_name: str,
    dataset_config: str,
    splits: List[str],
    num_processes: int = DEFAULT_NUM_PROCESSES
) -> Optional[Dict[str, Any]]:
    """
    Loads a dataset, processes specified splits using dataset_manipulation.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        dataset_config (str): Configuration of the dataset.
        splits (List[str]): List of splits to process (e.g., ['train', 'test']).
        num_processes (int): Number of parallel processes to use.

    Returns:
        Optional[Dict[str, Any]]: Dictionary mapping split names to processed data, or None on failure.
    """
    logging.info(f"Starting dataset loading ({dataset_name}, config: {dataset_config}) and manipulation...")
    processed_data = {}
    try:
        dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
        logging.info(f"Dataset loaded. Processing splits: {splits}")

        for split in splits:
            if split not in dataset:
                logging.warning(f"Split '{split}' not found in dataset. Skipping.")
                continue
            logging.info(f"Processing split '{split}'...")
            # Assuming ds_man.process_mp returns the processed data for the split
            processed_data[split] = ds_man.process_mp(
                dataset[split], num_processes=num_processes
            )
            logging.info(f"Finished processing split '{split}'.")

        if not processed_data:
             logging.error("No splits were successfully processed.")
             return None

        return processed_data

    except Exception as e:
        logging.error(f"Error during dataset loading or processing: {e}")
        return None


def convert_datasets_to_text(
    processed_data: Dict[str, Any],
    num_processes: int = DEFAULT_NUM_PROCESSES
) -> Optional[Dict[str, Any]]:
    """
    Converts processed dataset splits into text format using dataset_to_text.

    Args:
        processed_data (Dict[str, Any]): Dictionary of processed data per split.
        num_processes (int): Number of parallel processes to use.

    Returns:
        Optional[Dict[str, Any]]: Dictionary mapping split names to text data, or None on failure.
    """
    logging.info("Starting dataset to text conversion...")
    text_data = {}
    try:
        for split, data in processed_data.items():
            logging.info(f"Converting split '{split}' to text...")
            # Assuming ds2txt.ds_to_text returns the text data for the split
            text_data[split] = ds2txt.ds_to_text(data, num_processes)
            logging.info(f"Finished converting split '{split}' to text.")

        if not text_data:
             logging.error("No text data was generated.")
             return None
        return text_data
    except Exception as e:
        logging.error(f"Error during text conversion: {e}")
        return None


def generate_samples_from_texts(
    text_data: Dict[str, Any],
    tokenizer: Any,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    max_tokens_per_sample: int = DEFAULT_MAX_TOKENS_PER_SAMPLE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    num_processes: int = DEFAULT_NUM_PROCESSES,
) -> Optional[Dict[str, List[str]]]:
    """
    Generates text samples from text data using text_to_sample.

    Args:
        text_data (Dict[str, Any]): Dictionary of text data per split.
        tokenizer (Any): The tokenizer object.
        total_samples (int): Target number of samples per split.
        max_tokens_per_sample (int): Maximum tokens allowed per sample.
        chunk_size (int): Chunk size for processing.
        num_processes (int): Number of parallel processes.

    Returns:
        Optional[Dict[str, List[str]]]: Dictionary mapping split names to lists of samples, or None on failure.
    """
    logging.info("Starting sample generation from text...")
    samples_data = {}
    try:
        for split, texts in text_data.items():
            logging.info(f"Generating samples for split '{split}'...")
            # Assuming text2sample.distribute_processing returns a list of samples
            samples_data[split] = text2sample.distribute_processing(
                texts,
                total_samples=total_samples,
                tokenizer=tokenizer,
                max_tokens_per_sample=max_tokens_per_sample,
                chunk_size=chunk_size,
                num_processes=num_processes,
            )
            logging.info(f"Finished sample generation for split '{split}'. Found {len(samples_data[split])} samples.")

        if not samples_data:
             logging.error("No samples were generated.")
             return None
        return samples_data
    except Exception as e:
        logging.error(f"Error during sample generation: {e}")
        return None


def add_noise_to_samples(
    samples_data: Dict[str, List[str]],
    noise_prob: float = DEFAULT_NOISE_PROB,
    num_processes: int = DEFAULT_NUM_PROCESSES
) -> Optional[Dict[str, Any]]:
    """
    Adds noise to the generated text samples using noise_generator.

    Args:
        samples_data (Dict[str, List[str]]): Dictionary of samples per split.
        noise_prob (float): Probability of applying noise to each element.
        num_processes (int): Number of parallel processes.

    Returns:
        Optional[Dict[str, Any]]: Dictionary mapping split names to noised data, or None on failure.
    """
    logging.info(f"Starting adding noise to samples (probability: {noise_prob})...")
    noised_data = {}
    try:
        for split, samples in samples_data.items():
            logging.info(f"Adding noise to split '{split}'...")
            # Assuming noiseGen.process_mp returns the noised data for the split
            noised_data[split] = noiseGen.process_mp(
                samples, noise_prob=noise_prob, num_processes=num_processes
            )
            logging.info(f"Finished adding noise to split '{split}'.")

        if not noised_data:
             logging.error("No noised data was generated.")
             return None
        return noised_data
    except Exception as e:
        logging.error(f"Error during noise addition: {e}")
        return None


def format_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Formats a single entry (containing 'text' and 'target' keys from noising)
    into the final Prompt/Train/Target structure.

    Args:
        entry (Dict[str, Any]): A dictionary expected to have 'text' (noised)
                                and 'target' (original) keys.

    Returns:
        Dict[str, str]: A dictionary with "Prompt", "Train", and "Target" keys.
    """
    return {
        "Prompt": PROMPT_TEXT,
        "Train": entry.get("text", ""),    # Use .get for safety
        "Target": entry.get("target", ""), # Use .get for safety
    }


def create_prompt_dataset(
    noised_data: Dict[str, Any],
    num_processes: int = DEFAULT_NUM_PROCESSES
) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """
    Creates the final prompt dataset structure using multiprocessing.

    Args:
        noised_data (Dict[str, Any]): Dictionary of noised data per split.
        num_processes (int): Number of parallel processes.

    Returns:
        Optional[Dict[str, List[Dict[str, str]]]]: Dictionary mapping split names to
                                                   lists of formatted entries, or None on failure.
    """
    logging.info("Creating final prompt dataset structure...")
    prompt_dataset = {}
    try:
        # Use try-finally to ensure pool closure
        pool = mp.Pool(processes=num_processes)
        try:
            for split, data in noised_data.items():
                logging.info(f"Formatting entries for split '{split}'...")
                # Map the format_entry function over the data for the current split
                prompt_dataset[split] = pool.map(format_entry, data)
                logging.info(f"Finished formatting entries for split '{split}'.")
        finally:
            pool.close()
            pool.join()

        if not prompt_dataset:
             logging.error("Prompt dataset creation resulted in empty data.")
             return None
        return prompt_dataset
    except Exception as e:
        logging.error(f"Error during prompt dataset creation: {e}")
        return None


def save_dataset_to_file(dataset: Dataset, filename: str) -> bool:
    """
    Saves a Hugging Face Dataset object to a file using pickle.

    Args:
        dataset (Dataset): The Hugging Face dataset to save.
        filename (str): The path to the output pickle file.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    logging.info(f"Saving dataset to {filename}...")
    try:
        with open(filename, "wb") as f:
            pickle.dump(dataset, f)
        logging.info(f"Successfully saved dataset to {filename}.")
        return True
    except (IOError, pickle.PicklingError) as e:
        logging.error(f"Failed to save dataset to {filename}: {e}")
        return False


def main():
    """
    Main function to run the entire dataset generation pipeline.
    """
    logging.info("--- Starting Dataset Generation Pipeline ---")

    # --- 1. Load and process datasets ---
    processed_data = load_and_process_datasets(
        DATASET_NAME, DATASET_CONFIG, DEFAULT_SPLITS, DEFAULT_NUM_PROCESSES
    )
    if not processed_data:
        logging.critical("Failed to load or process initial dataset. Exiting.")
        exit(1)

    # --- 2. Convert processed datasets to text ---
    text_data = convert_datasets_to_text(processed_data, DEFAULT_NUM_PROCESSES)
    if not text_data:
        logging.critical("Failed to convert dataset to text. Exiting.")
        exit(1)

    # --- 3. Load tokenizer ---
    try:
        logging.info(f"Loading tokenizer: {TOKENIZER_ID}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load tokenizer '{TOKENIZER_ID}': {e}. Exiting.")
        exit(1)

    # --- 4. Generate samples from texts ---
    samples_data = generate_samples_from_texts(
        text_data,
        tokenizer,
        DEFAULT_TOTAL_SAMPLES,
        DEFAULT_MAX_TOKENS_PER_SAMPLE,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_NUM_PROCESSES,
    )
    if not samples_data:
        logging.critical("Failed to generate samples from text. Exiting.")
        exit(1)

    # --- 5. Post-process samples (ensure they end with a full stop) ---
    logging.info("Post-processing generated samples...")
    for split in samples_data:
        logging.debug(f"Post-processing samples for split '{split}'...")
        processed_count = 0
        for idx, text_sample in enumerate(samples_data[split]):
            # This aims to remove potential sentence fragments at the end of a sample
            # by splitting on ". ", dropping the last part, rejoining, and adding "."
            parts = text_sample.split(". ")
            if len(parts) > 1:
                 samples_data[split][idx] = ". ".join(parts[:-1]) + "."
                 processed_count += 1
            # else: keep the original sample if it doesn't contain ". "
        logging.debug(f"Finished post-processing {processed_count} samples for split '{split}'.")
    logging.info("Finished post-processing all samples.")


    # --- 6. Add noise to samples ---
    noised_data = add_noise_to_samples(
        samples_data, DEFAULT_NOISE_PROB, DEFAULT_NUM_PROCESSES
    )
    if not noised_data:
        logging.critical("Failed to add noise to samples. Exiting.")
        exit(1)

    # --- 7. Create the prompt dataset ---
    prompt_dataset_map = create_prompt_dataset(noised_data, DEFAULT_NUM_PROCESSES)
    if not prompt_dataset_map:
        logging.critical("Failed to create prompt dataset. Exiting.")
        exit(1)

    # --- 8. Convert to Hugging Face Dataset and Save ---
    logging.info("Converting final data to Hugging Face Datasets and saving...")
    for split, data_list in prompt_dataset_map.items():
        logging.info(f"Processing final dataset for split '{split}'...")
        try:
            # Directly create dataset from the list of dictionaries
            hf_dataset = Dataset.from_list(data_list)
            output_filename = OUTPUT_FILENAME_TEMPLATE.format(split=split)
            save_dataset_to_file(hf_dataset, output_filename)
        except Exception as e:
            logging.error(f"Failed to create or save dataset for split '{split}': {e}")

    logging.info("--- Dataset Generation Pipeline Finished ---")


if __name__ == "__main__":
    # Set start method for multiprocessing if needed (can be important on some OS)
    try:
        if mp.get_start_method(allow_none=True) != 'fork': # 'fork' is often default on Linux
             mp.set_start_method("spawn", force=True) # 'spawn' or 'forkserver' might be needed
             logging.info(f"Set multiprocessing start method to '{mp.get_start_method()}'.")
    except Exception as e:
         logging.warning(f"Could not set multiprocessing start method: {e}")

    main()
