import multiprocessing
import logging
from typing import List

# --- Constants ---
DEFAULT_NUM_PROCESSES = 4 # Default number of processes for this specific task

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def ds_to_text_chunk(processed_ds_chunk: List[str]) -> str:
    """
    Concatenates a list of text strings into a single string,
    with each original string separated by a newline character.

    Args:
        processed_ds_chunk (List[str]): A list of text strings.

    Returns:
        str: A single string containing all input strings joined by newlines.
    """
    # Efficiently join the list of strings with a newline separator
    return "\n".join(processed_ds_chunk) + "\n" # Add trailing newline


def ds_to_text(processed_ds: List[str], num_processes: int = DEFAULT_NUM_PROCESSES) -> str:
    """
    Converts a list of processed text strings into a single large text block
    using multiprocessing.

    Args:
        processed_ds (List[str]): The list of text strings to concatenate.
        num_processes (int): The number of worker processes to use.

    Returns:
        str: A single string containing all input strings concatenated,
             preserving the newline separation from chunks. Returns an empty
             string if input is empty or processing fails.
    """
    all_text = ""
    try:
        num_samples = len(processed_ds)
        if num_samples == 0:
            logging.warning("Input processed_ds is empty. Returning empty string.")
            return ""

        # Adjust num_processes if it exceeds the number of samples
        if num_processes > num_samples:
            num_processes = num_samples
            logging.info(f"Reduced number of processes to {num_processes} (number of samples).")

        # Calculate chunk size for roughly even distribution
        chunk_size = (num_samples + num_processes - 1) // num_processes
        chunks = [
            processed_ds[i : min(i + chunk_size, num_samples)]
            for i in range(0, num_samples, chunk_size)
        ]
        logging.info(f"Split input data into {len(chunks)} chunks for {num_processes} processes.")

        results = []
        # Use try-finally to ensure pool closure
        pool = multiprocessing.Pool(processes=num_processes)
        try:
            results = pool.map(ds_to_text_chunk, chunks)
            logging.info("Finished processing text chunks in parallel.")
        except Exception as e:
            logging.error(f"Error during multiprocessing pool execution in ds_to_text: {e}")
            return "" # Return empty string on pool error
        finally:
            pool.close()
            pool.join()

        # Join the results from each chunk. Since ds_to_text_chunk adds newlines,
        # joining with an empty string preserves this separation.
        all_text = "".join(results)
        logging.info(f"Concatenated results. Total text length: {len(all_text)} chars.")

    except Exception as e:
        logging.error(f"An unexpected error occurred in ds_to_text: {e}")
        return "" # Return empty string on unexpected errors

    return all_text
