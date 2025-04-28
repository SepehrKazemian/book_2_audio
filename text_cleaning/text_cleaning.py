import pickle
import logging
from typing import List, Optional

# Use absolute imports from project root
try:
    from text_cleaning import line_structure as ls
    from text_cleaning import line_cleanup as lc
except ImportError:
    logging.error("Failed to import line_structure or line_cleanup. Make sure they are accessible from the project root.")
    # Define dummy functions or exit if critical
    class ls:
        @staticmethod
        def count_characters_per_line(*args, **kwargs): return []
        @staticmethod
        def calculate_line_length_stats(*args, **kwargs): return None
        @staticmethod
        def classify_lines(*args, **kwargs): return []
    class lc:
        @staticmethod
        def filter_unwanted_lines(*args, **kwargs): return args[1], args[0] # Return original lists
        @staticmethod
        def paragraph_line_concat(*args, **kwargs): return args[1]
        @staticmethod
        def paragraph_line_cleanup(*args, **kwargs): return args[0]
        @staticmethod
        def bullet_point_cleanup(*args, **kwargs): return args[0]
    # exit(1) # Or raise error

# --- Constants ---
# !! IMPORTANT: Consider making paths configurable !!
# Input should be the output of the pdf_extraction step
INPUT_PICKLE_PATH = "data/extracted_book_text.pkl"
# Heuristic factor to determine short line threshold based on mean line length
SHORT_LINE_THRESHOLD_FACTOR = 0.5
# Optional: Define an output path if this script should save its results
# OUTPUT_CLEANED_PICKLE_PATH = "data/cleaned_book_text.pkl"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> Optional[List[str]]:
    """
    Main function to orchestrate the text cleaning process.

    Loads text from a pickle file, analyzes line structure, classifies lines,
    and applies various cleanup functions from line_cleanup module.

    Returns:
        Optional[List[str]]: A list of cleaned text lines, or None if an error occurs.
    """
    logging.info("--- Starting Text Cleaning Process ---")

    # --- 1. Load Data ---
    try:
        logging.info(f"Loading book data from: {INPUT_PICKLE_PATH}")
        with open(INPUT_PICKLE_PATH, "rb") as file:
            # Assuming the pickle contains a dictionary {page_num: [text, prob]}
            book_dict = pickle.load(file)
        logging.info(f"Loaded data for {len(book_dict)} pages.")
    except FileNotFoundError:
        logging.error(f"Input pickle file not found: {INPUT_PICKLE_PATH}. Exiting.")
        return None
    except (pickle.UnpicklingError, IOError, EOFError) as e:
        logging.error(f"Failed to load or unpickle data from {INPUT_PICKLE_PATH}: {e}. Exiting.")
        return None
    except Exception as e:
         logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
         return None

    # --- 2. Convert to Text and Initial Split ---
    # Concatenate text from all pages, assuming structure {page: [text, prob]}
    try:
        # Sort by page number (assuming keys are numeric) before joining for consistency
        sorted_pages = sorted(book_dict.items())
        full_text = "\n".join([item[1][0] for item in sorted_pages if isinstance(item[1], list) and len(item[1]) > 0 and isinstance(item[1][0], str)])
        if not full_text:
             logging.error("No text could be extracted from the loaded dictionary.")
             return None
        lines = full_text.split("\n") # Initial split for analysis
        logging.info(f"Converted dictionary to text ({len(lines)} initial lines).")
    except Exception as e:
        logging.error(f"Error converting loaded dictionary to text: {e}", exc_info=True)
        return None

    # --- 3. Character/Structure Analysis ---
    logging.info("Analyzing line structure (character counts, stats)...")
    char_counts = ls.count_characters_per_line(full_text) # Use full_text here
    if not char_counts:
        logging.error("Character count analysis returned empty results.")
        return None

    stats = ls.calculate_line_length_stats(char_counts)
    if stats is None:
        logging.warning("Could not calculate line length statistics. Proceeding without threshold-based classification.")
        # Set a default threshold or handle classification differently
        short_line_threshold = 10 # Arbitrary default if stats fail
    else:
        mean_len, std_dev = stats
        # Determine threshold for short lines (heuristic)
        short_line_threshold = mean_len * SHORT_LINE_THRESHOLD_FACTOR
        logging.info(f"Using short line threshold: {short_line_threshold:.2f}")

    classifications = ls.classify_lines(lines, char_counts, short_line_threshold)
    if not classifications:
         logging.error("Line classification failed.")
         return None

    # --- 4. Line-Based Cleanup ---
    logging.info("Performing line-based cleanup (filtering, concatenation)...")
    # Create check_line: mark lines *not* classified as "Regular Line" for checking
    # These are lines we might want to filter or treat specially during concatenation
    check_line = [
        "check" if cls[1] != "Regular Line" else "uncheck"
        for cls in classifications
    ]

    # Filter unwanted lines (only those marked 'check')
    lines, check_line = lc.filter_unwanted_lines(check_line, lines)
    if len(lines) != len(check_line): # Sanity check after filtering
         logging.error("Mismatch length after filter_unwanted_lines. Aborting.")
         return None

    # Concatenate paragraph lines (respecting 'check' markers)
    lines = lc.paragraph_line_concat(check_line, lines)

    # --- 5. Paragraph/Final Cleanup ---
    logging.info("Performing final cleanup (whitespace, bullets)...")
    # Cleanup whitespace, tabs, and remove empty lines resulting from previous steps
    lines = lc.paragraph_line_cleanup(lines)
    # Cleanup isolated bullet points
    lines = lc.bullet_point_cleanup(lines)

    logging.info(f"--- Text Cleaning Process Finished. Returning {len(lines)} cleaned lines. ---")
    return lines


if __name__ == "__main__":
    cleaned_lines = main()
    if cleaned_lines:
        logging.info("Cleaning process completed successfully.")
        # Optional: Save the cleaned lines to a new file or print a sample
        # print("\nSample of cleaned lines:")
        # for line in cleaned_lines[:20]:
        #     print(line)
        # with open("cleaned_text.txt", "w") as f:
        #     f.write("\n".join(cleaned_lines))
        # logging.info("Saved cleaned text to cleaned_text.txt")
    else:
        logging.error("Cleaning process failed.")
