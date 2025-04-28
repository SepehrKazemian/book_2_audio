import random
import string
import re
import multiprocessing
import logging
from typing import List, Dict

# --- Constants ---
# List of special (often non-printable or whitespace) characters to potentially add
SPECIAL_CHARS = ["\x0a", "\x0c", "\x0b", "\x07", "\x0d", "\x09", "\x08"]
# Available types of noise operations
NOISE_TYPES = ["remove", "add", "swap", "special"]
# Regex to find and remove dataset markers (e.g., _START_ARTICLE_)
MARKER_REGEX = r"_[A-Z]+(_[A-Z]+)*_"
# Default probability for applying noise to a word
DEFAULT_NOISE_PROB = 0.1
# Default number of processes for multiprocessing
DEFAULT_NUM_PROCESSES = 12

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === Noise Functions ===

def add_special_character(word: str) -> str:
    """Adds a random special character from SPECIAL_CHARS to the beginning of a word."""
    if word: # Check if word is not empty
        special_char = random.choice(SPECIAL_CHARS)
        return special_char + word
    return word

def remove_random_letter(word: str) -> str:
    """Removes a single random letter from the word."""
    if len(word) > 1:
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1 :]
    # Return word as is if it's too short or empty
    return word

def add_random_character(word: str) -> str:
    """Adds a random lowercase ASCII letter at a random position in the word."""
    idx = random.randint(0, len(word)) # Allow insertion at the end
    char = random.choice(string.ascii_lowercase)
    return word[:idx] + char + word[idx:]

def swap_random_characters(word: str) -> str:
    """Swaps two random characters within the word."""
    if len(word) > 1:
        # Ensure two different indices are chosen for swapping
        idx1 = random.randint(0, len(word) - 1)
        idx2 = random.randint(0, len(word) - 1)
        # Retry if indices are the same (simple approach for low collision probability)
        while idx1 == idx2 and len(word) > 1: # Check len > 1 again in case word becomes short
             idx2 = random.randint(0, len(word) - 1)

        word_list = list(word)
        word_list[idx1], word_list[idx2] = word_list[idx2], word_list[idx1]
        return "".join(word_list)
    return word

def introduce_noise(word: str, noise_type: str) -> str:
    """Applies a specified type of noise to a single word."""
    noise_functions = {
        "remove": remove_random_letter,
        "add": add_random_character,
        "swap": swap_random_characters,
        "special": add_special_character,
    }
    # Get the function from the dictionary, default to returning the word if type is unknown
    func = noise_functions.get(noise_type, lambda w: w)
    return func(word)

def apply_noise(text: str, noise_prob: float = DEFAULT_NOISE_PROB) -> str:
    """
    Applies random noise to words in a text string based on a probability.

    Args:
        text (str): The input text string.
        noise_prob (float): The probability (0.0 to 1.0) of applying noise to each word.

    Returns:
        str: The text with noise potentially applied to some words.
    """
    # Using regex to find words might be more robust than split() if punctuation is attached
    # words = re.findall(r'\b\w+\b', text) # Example: find word boundaries
    # For simplicity, sticking to split() as per original code
    words = text.split()
    noisy_words = []
    for word in words:
        if random.random() < noise_prob:
            chosen_noise_type = random.choice(NOISE_TYPES)
            noisy_word = introduce_noise(word, noise_type=chosen_noise_type)
            noisy_words.append(noisy_word)
        else:
            noisy_words.append(word)
    return " ".join(noisy_words)


def remove_markers(text: str) -> str:
    """Removes dataset markers (like _START_ARTICLE_) from the text."""
    # Replace all occurrences of the marker pattern with a space to avoid merging words
    cleaned_text = re.sub(MARKER_REGEX, " ", text)
    # Optionally, clean up multiple spaces resulting from replacement
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


# === Processing Functions ===

def process_line(line: str, noise_prob: float) -> Dict[str, str]:
    """
    Processes a single line of text: removes markers and applies noise.

    Args:
        line (str): The input line of text.
        noise_prob (float): The probability for applying noise.

    Returns:
        Dict[str, str]: A dictionary with 'text' (noised) and 'target' (original) keys.
    """
    original_text = line # Keep the original line as the target
    cleaned_text = remove_markers(original_text)
    noisy_text = apply_noise(cleaned_text, noise_prob)
    return {"text": noisy_text, "target": original_text}


def process_chunk(chunk: List[str], noise_prob: float) -> List[Dict[str, str]]:
    """
    Applies the `process_line` function to each line in a chunk of text lines.

    Args:
        chunk (List[str]): A list of text lines.
        noise_prob (float): The noise probability to pass to `process_line`.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each representing a processed line.
    """
    return [process_line(line, noise_prob) for line in chunk]


def process_mp(
    data: List[str],
    noise_prob: float = DEFAULT_NOISE_PROB,
    num_processes: int = DEFAULT_NUM_PROCESSES
) -> List[Dict[str, str]]:
    """
    Applies noise generation to a list of text data in parallel using multiprocessing.

    Args:
        data (List[str]): The input list of text strings.
        noise_prob (float): The probability of applying noise to each word.
        num_processes (int): The number of worker processes.

    Returns:
        List[Dict[str, str]]: A list of dictionaries {'text': noised_text, 'target': original_text}.
                              Returns an empty list if processing fails.
    """
    processed_data = []
    try:
        num_samples = len(data)
        if num_samples == 0:
            logging.warning("Input data list is empty. Returning empty list.")
            return []

        # Adjust num_processes if it exceeds the number of samples
        if num_processes > num_samples:
            num_processes = num_samples
            logging.info(f"Reduced number of processes to {num_processes} (number of samples).")

        # Calculate chunk size for roughly even distribution
        chunk_size = (num_samples + num_processes - 1) // num_processes
        chunks = [
            data[i : min(i + chunk_size, num_samples)]
            for i in range(0, num_samples, chunk_size)
        ]
        logging.info(f"Split data into {len(chunks)} chunks for {num_processes} processes.")

        processed_chunks = []
        # Use try-finally to ensure pool closure
        pool = multiprocessing.Pool(processes=num_processes)
        try:
            # Prepare arguments for starmap: list of tuples [(chunk1, noise_prob), (chunk2, noise_prob), ...]
            starmap_args = [(chunk, noise_prob) for chunk in chunks]
            processed_chunks = pool.starmap(process_chunk, starmap_args)
            logging.info("Finished processing noise chunks in parallel.")
        except Exception as e:
            logging.error(f"Error during multiprocessing pool execution in noise generation: {e}")
            return [] # Return empty list on pool error
        finally:
            pool.close()
            pool.join()

        # Concatenate the results from processed chunks
        processed_data = [item for sublist in processed_chunks for item in sublist]
        logging.info(f"Concatenated noise results. Total processed items: {len(processed_data)}")

    except Exception as e:
        logging.error(f"An unexpected error occurred in noise generation process_mp: {e}")
        return [] # Return empty list on unexpected errors

    return processed_data
