import numpy as np
import re
import logging
from typing import List, Tuple, Optional

# --- Constants ---
DOT_DASH_THRESHOLD = 5 # Threshold for classifying line based on dot/dash count

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pre-compiled Regex ---
# Matches lines that appear to be solely Roman numerals (e.g., "I.", "II", "xvii")
ROMAN_NUMERAL_PATTERN = re.compile(
    r"^\s*(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?\s*$",
    re.IGNORECASE # Match lowercase Roman numerals too
)


def count_characters_per_line(text: str) -> List[int]:
    """
    Counts the number of characters in each line of the input text.

    Args:
        text (str): The input text, potentially multi-line.

    Returns:
        List[int]: A list where each element is the character count of the corresponding line.
                   Returns an empty list if the input text is empty.
    """
    if not text:
        logging.warning("Input text is empty in count_characters_per_line.")
        return []
    lines = text.split("\n")
    char_counts = [len(line) for line in lines]
    logging.debug(f"Counted characters for {len(char_counts)} lines.")
    return char_counts


def calculate_line_length_stats(line_lengths: List[int]) -> Optional[Tuple[float, float]]:
    """
    Calculates the mean and standard deviation of non-zero line lengths.

    Args:
        line_lengths (List[int]): A list of character counts per line.

    Returns:
        Optional[Tuple[float, float]]: A tuple containing (mean_length, std_dev),
                                       or None if no valid lengths are found.
    """
    # Filter out zero-length lines before calculating stats
    valid_lengths = [length for length in line_lengths if length > 0]

    if not valid_lengths:
        logging.warning("No valid line lengths found to calculate statistics.")
        return None

    try:
        mean_length = np.mean(valid_lengths)
        std_dev = np.std(valid_lengths)
        logging.info(f"Calculated line length stats: Mean={mean_length:.2f}, StdDev={std_dev:.2f}")
        return mean_length, std_dev
    except Exception as e:
        logging.error(f"Error calculating line length statistics: {e}", exc_info=True)
        return None


def classify_lines(
    lines: List[str],
    line_lengths: List[int],
    short_line_threshold: float
) -> List[Tuple[int, str]]:
    """
    Classifies lines based on length, content (Roman numerals), and dot/dash count.

    Args:
        lines (List[str]): The list of text lines.
        line_lengths (List[int]): Corresponding list of character counts for each line.
        short_line_threshold (float): Lines with length below this are classified differently.

    Returns:
        List[Tuple[int, str]]: A list of tuples, where each tuple contains the original
                               line length and its classification string
                               (e.g., "Roman Numeral", "Likely Short/Header", "Regular Line").
                               Returns an empty list if input lists are mismatched or empty.
    """
    if not lines or not line_lengths or len(lines) != len(line_lengths):
        logging.error("Input lists are empty or mismatched in classify_lines.")
        return []

    classifications = []
    logging.debug(f"Classifying {len(lines)} lines using threshold: {short_line_threshold:.2f}")

    for i, length in enumerate(line_lengths):
        line_content = lines[i]
        stripped_line = line_content.strip()
        classification = "Regular Line" # Default classification

        # Count dots and dashes
        dot_dash_count = line_content.count(".") + line_content.count("-")

        # --- Classification Logic ---
        if not stripped_line: # Empty or whitespace-only line
             classification = "Empty Line"
        elif ROMAN_NUMERAL_PATTERN.fullmatch(stripped_line):
            classification = "Roman Numeral"
        # Classify short lines or lines with many dots/dashes (potential separators/noise)
        elif length < short_line_threshold or dot_dash_count > DOT_DASH_THRESHOLD:
             classification = "Likely Short/Header" # Renamed from "New Line"
        # else: # All other lines are considered "Regular Line" (default)
        #    pass

        classifications.append((length, classification))
        logging.log(logging.DEBUG - 1, f"Line {i} (len={length}): Classified as '{classification}' - Content: '{line_content[:50]}...'")

    return classifications
