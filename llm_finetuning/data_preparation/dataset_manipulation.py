import multiprocessing
import random
import logging
from typing import List
from datasets import Dataset # Import for type hinting

# --- Constants ---
REMOVAL_PROBABILITY = 0.7 # Probability to remove a section or article
START_SECTION_TAG = "_START_SECTION_"
START_ARTICLE_TAG = "_START_ARTICLE_"
START_PARAGRAPH_TAG = "_START_PARAGRAPH_"
DEFAULT_NUM_PROCESSES = 12 # Match dataset_main or make configurable

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def remove_random_sections_and_articles(text: str) -> str:
    """
    Processes text to randomly remove sections or articles based on specific tags.

    It identifies lines starting sections and articles, randomly selects a portion
    (defined by REMOVAL_PROBABILITY) to remove, and removes the content from
    the start tag up to the next paragraph start tag.

    Args:
        text (str): The input text containing section/article/paragraph tags.

    Returns:
        str: The text with randomly selected sections/articles removed.
    """
    lines = text.split("\n")
    sections_indices = []
    articles_indices = []

    # Find all _START_SECTION_ and _START_ARTICLE_ tags
    for idx, line in enumerate(lines):
        if START_SECTION_TAG in line:
            sections_indices.append(idx)
        elif START_ARTICLE_TAG in line:
            articles_indices.append(idx)

    # Randomly select a fraction of sections and articles to remove
    sections_to_remove = set(
        idx for idx in sections_indices if random.random() < REMOVAL_PROBABILITY
    )
    articles_to_remove = set(
        idx for idx in articles_indices if random.random() < REMOVAL_PROBABILITY
    )
    logging.debug(f"Identified {len(sections_indices)} sections, removing {len(sections_to_remove)}.")
    logging.debug(f"Identified {len(articles_indices)} articles, removing {len(articles_to_remove)}.")

    # Reconstruct the text, skipping lines within the selected sections/articles
    new_text_lines = []
    skip_lines = False # Flag to indicate if we are currently inside a section/article to be removed
    for idx, line in enumerate(lines):
        # Check if the current line marks the start of a section/article to remove
        if idx in sections_to_remove or idx in articles_to_remove:
            skip_lines = True
            logging.debug(f"Starting skip at line {idx}: {line[:50]}...")
        # Check if the current line marks the start of a paragraph, which ends the skipping
        elif START_PARAGRAPH_TAG in line:
            if skip_lines:
                 logging.debug(f"Ending skip at line {idx}: {line[:50]}...")
            skip_lines = False

        # Append the line if we are not currently skipping
        # Also append the _START_PARAGRAPH_ line itself, even if skipping was just turned off
        if not skip_lines:
            new_text_lines.append(line)
        # If we just stopped skipping because of a paragraph tag, make sure it's added
        elif START_PARAGRAPH_TAG in line:
             new_text_lines.append(line)


    return "\n".join(new_text_lines)


def process_chunk(chunk: Dataset) -> Dataset:
    """
    Applies the `remove_random_sections_and_articles` function to the 'text'
    field of each example in a dataset chunk.

    Args:
        chunk (Dataset): A Hugging Face Dataset object (or a shard).

    Returns:
        Dataset: The processed dataset chunk with modified 'text' field.
    """
    try:
        # Apply the preprocessing function using map
        # The [1:] slice in the original code might have been intended to remove a
        # leading newline potentially left by joining in remove_random_sections_and_articles.
        # Using .lstrip('\n') might be more robust if that's the goal.
        # Keeping original logic for now, but adding note.
        # TODO: Investigate if `[1:]` is necessary or if `lstrip('\n')` is better.
        processed_chunk = chunk.map(
            lambda example: {
                "text": remove_random_sections_and_articles(example["text"])[1:]
            }
        )
        return processed_chunk
    except Exception as e:
        logging.error(f"Error processing chunk in process_chunk: {e}")
        # Return the original chunk or an empty one depending on desired error handling
        return chunk # Example: return original chunk on error


def process_mp(data: Dataset, num_processes: int = DEFAULT_NUM_PROCESSES) -> List[str]:
    """
    Processes a Hugging Face Dataset in parallel using multiprocessing.

    Splits the dataset into chunks and applies `process_chunk` to each using a pool
    of worker processes.

    Args:
        data (Dataset): The input Hugging Face Dataset.
        num_processes (int): The number of worker processes to use.

    Returns:
        List[str]: A list containing the processed 'text' field from all examples
                   in the dataset. Returns an empty list if processing fails.
    """
    processed_data_list = []
    try:
        num_samples = len(data)
        if num_samples == 0:
            logging.warning("Input dataset is empty. Returning empty list.")
            return []

        # Calculate chunk size for roughly even distribution
        chunk_size = (num_samples + num_processes - 1) // num_processes
        # Create chunks using dataset's select method
        chunks = [
            data.select(range(i, min(i + chunk_size, num_samples)))
            for i in range(0, num_samples, chunk_size)
        ]
        logging.info(f"Split dataset into {len(chunks)} chunks for {num_processes} processes.")

        # Create a pool of worker processes
        processed_chunks = []
        # Use try-finally to ensure pool closure
        pool = multiprocessing.Pool(processes=num_processes)
        try:
            processed_chunks = pool.map(process_chunk, chunks)
            logging.info("Finished processing chunks in parallel.")
        except Exception as e:
             logging.error(f"Error during multiprocessing pool execution: {e}")
             # Decide on error handling: return partial results, empty list, raise error?
             return [] # Example: return empty list on pool error
        finally:
            pool.close()
            pool.join()

        # Concatenate the processed data from chunks into a single list of texts
        # Assumes process_chunk returns a Dataset object
        processed_data_list = [
            item["text"] for chunk in processed_chunks for item in chunk
        ]
        logging.info(f"Concatenated results. Total processed items: {len(processed_data_list)}")

    except Exception as e:
        logging.error(f"An unexpected error occurred in process_mp: {e}")
        return [] # Return empty list on unexpected errors

    return processed_data_list
