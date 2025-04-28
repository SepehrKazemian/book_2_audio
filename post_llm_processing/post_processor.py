import pickle
import logging
from typing import Dict, Any, List, Optional

# --- Constants ---
# !! IMPORTANT: Consider making these paths configurable !!
# Assumes the input is the output from the LLM correction step
DEFAULT_INPUT_PICKLE = "data/text_correction_results.pkl" # Assuming output from llm_correction is saved here
DEFAULT_OUTPUT_PICKLE = "data/final_processed_book.pkl" # Name for the final output after this step
# Keys (page numbers/indices) to remove from the processed data
# These were hardcoded in the notebook, make sure they are correct for your book
KEYS_TO_REMOVE = [0, 2, 3]

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def remove_specific_pages(
    input_pickle_path: str = DEFAULT_INPUT_PICKLE,
    output_pickle_path: str = DEFAULT_OUTPUT_PICKLE,
    keys_to_remove: List[Any] = KEYS_TO_REMOVE
) -> bool:
    """
    Loads processed book data, removes specified pages/keys, and saves the result.

    Args:
        input_pickle_path (str): Path to the input pickle file (output from LLM correction).
                                 Expected format: Dict[page_key, Any] or List[Dict[str, Any]]
        output_pickle_path (str): Path to save the final processed data.
        keys_to_remove (List[Any]): List of keys (e.g., page numbers) to remove.

    Returns:
        bool: True if processing and saving were successful, False otherwise.
    """
    logging.info(f"Starting post-LLM processing: Removing specific pages/keys.")
    logging.info(f"Loading data from: {input_pickle_path}")

    try:
        with open(input_pickle_path, "rb") as file:
            # Load the data. The structure from text_llm_correction was List[Optional[Dict[str, Any]]]
            # We need to convert this back to a dictionary keyed by page/batch index if possible,
            # or handle the list structure directly.
            # Let's assume for now the goal is to filter the list based on 'batch_index' matching keys_to_remove.
            data_list = pickle.load(file)

        if not isinstance(data_list, list):
             logging.error(f"Loaded data from {input_pickle_path} is not a list as expected. Type: {type(data_list)}")
             return False

        logging.info(f"Loaded {len(data_list)} entries.")

        # Filter the list, keeping entries whose 'batch_index' is NOT in keys_to_remove
        # Note: The original notebook deleted keys from a dictionary. This adapts to the list structure
        # produced by the refactored text_llm_correction.py.
        # We also filter out entries that might have failed during processing.
        filtered_data_list = [
            entry for entry in data_list
            if entry is not None and
               entry.get("status") == "success" and # Only keep successful entries
               entry.get("batch_index") not in keys_to_remove
        ]

        removed_count = len(data_list) - len(filtered_data_list)
        logging.info(f"Removed {removed_count} entries corresponding to keys {keys_to_remove} or failed status.")

        # Save the filtered data
        logging.info(f"Saving final processed data ({len(filtered_data_list)} entries) to: {output_pickle_path}")
        with open(output_pickle_path, "wb") as file:
            pickle.dump(filtered_data_list, file)
        logging.info("Successfully saved final processed data.")
        return True

    except FileNotFoundError:
        logging.error(f"Input pickle file not found: {input_pickle_path}")
        return False
    except (pickle.UnpicklingError, pickle.PicklingError, IOError, EOFError) as e:
        logging.error(f"Error during pickle loading/saving: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during post-processing: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logging.info("--- Running Post-LLM Processing Script ---")
    success = remove_specific_pages()
    if success:
        logging.info("Post-processing completed successfully.")
    else:
        logging.error("Post-processing failed.")
        exit(1)
    logging.info("--- Post-LLM Processing Script Finished ---")
