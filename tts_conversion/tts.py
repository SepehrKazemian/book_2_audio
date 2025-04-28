import torch
import multiprocessing as mp
import re
import os
import pickle
import logging
import glob
from pydub import AudioSegment
from typing import List, Optional

# Use absolute import from project root
try:
    from tts_conversion.processing_audio import process_text, split_text
except ImportError:
    logging.error("Failed to import from processing_audio.py. Make sure it's in the tts_conversion directory and the script is run from the project root.")
    # Define dummy functions or exit if critical
    def process_text(*args, **kwargs): raise NotImplementedError("processing_audio.py not found")
    def split_text(*args, **kwargs): raise NotImplementedError("processing_audio.py not found")

try:
    from TTS.api import TTS
except ImportError:
     logging.error("TTS library not found. Please install it: pip install TTS")
     # Define dummy class or exit if critical
     class TTS:
         def __init__(self, *args, **kwargs): raise NotImplementedError("TTS library not found")
         def tts_to_file(self, *args, **kwargs): raise NotImplementedError("TTS library not found")


# --- Constants ---
# !! IMPORTANT: Consider making paths configurable !!
# Input should likely be the output of the post-processing step
INPUT_PICKLE_FILE = "data/final_processed_book.pkl"
# Output files should ideally go into the data directory
OUTPUT_FILE_PREFIX = "data/output_part" # Prefix for intermediate files
FINAL_OUTPUT_WAV = "data/final_output.wav"
AUDIO_FADE_MS = 10 # Fade duration for concatenating audio segments
# Speaker WAV file should also be configurable/in data
SPEAKER_WAV_PATH = "data/wav_file.wav" # Assuming it's in data/

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def concatenate_audio_files(file_list: List[str], output_file: str) -> bool:
    """
    Concatenates a list of WAV audio files into a single output WAV file
    with short fades between segments.

    Args:
        file_list (List[str]): A list of paths to the WAV files to concatenate.
        output_file (str): The path for the final concatenated WAV file.

    Returns:
        bool: True if concatenation was successful, False otherwise.
    """
    if not file_list:
        logging.warning("No audio files provided for concatenation.")
        return False

    combined = AudioSegment.empty()
    logging.info(f"Concatenating {len(file_list)} audio files into {output_file}...")

    for filename in file_list:
        try:
            segment = AudioSegment.from_wav(filename)
            # Add fade in and fade out to each segment for smoother transitions
            if AUDIO_FADE_MS > 0:
                segment = segment.fade_in(AUDIO_FADE_MS).fade_out(AUDIO_FADE_MS)
            combined += segment
        except FileNotFoundError:
            logging.warning(f"Audio file not found, skipping: {filename}")
        except Exception as e:
            logging.error(f"Error processing audio file {filename}: {e}")
            # Optionally decide whether to stop or continue
            # return False # Example: stop on first error

    try:
        combined.export(output_file, format="wav")
        logging.info(f"Successfully exported concatenated audio to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to export combined audio file {output_file}: {e}")
        return False


def remove_page_numbers(text: str) -> str:
    """
    Removes lines that likely represent page numbers from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with potential page number lines removed.
    """
    # Pattern: digits at the start/end of a line, possibly surrounded by whitespace.
    # Handles cases like "123", "  45 ", "Title 67", "89 Chapter"
    # It's conservative to avoid removing numbers within sentences.
    pattern = re.compile(r"^\s*\d+\s*$|^\s*\d+\s+[^\n]*$|^[^\n]*\s+\d+\s*$", re.MULTILINE)
    # More aggressive pattern if needed: r"(^\d+\s*$)|(^\d+\s+)|(\s+\d+\s*$)"

    cleaned_text = re.sub(pattern, "", text)
    # Remove potential excess blank lines left after removal
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
    return cleaned_text.strip()


def load_text(input_pickle: str) -> Optional[str]:
    """
    Loads text data from a pickle file, concatenates text from pages,
    and removes page numbers.

    Args:
        input_pickle (str): Path to the input pickle file. Expected format:
                            Dict[page_num, [page_text, probability]].

    Returns:
        Optional[str]: The concatenated and cleaned text, or None on error.
    """
    try:
        logging.info(f"Loading text data from {input_pickle}...")
        with open(input_pickle, "rb") as file:
            data_dict = pickle.load(file)
        logging.info(f"Loaded data for {len(data_dict)} pages.")
    except FileNotFoundError:
        logging.error(f"Input pickle file not found: {input_pickle}")
        return None
    except (pickle.UnpicklingError, IOError, EOFError) as e:
        logging.error(f"Failed to load or unpickle data from {input_pickle}: {e}")
        return None

    full_text = ""
    for page_num, value in data_dict.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
            text_page = value[0]
            full_text += text_page + "\n\n" # Add double newline between pages
        else:
            logging.warning(f"Unexpected data format for page {page_num}: {value}. Skipping.")

    if not full_text:
        logging.warning("No text extracted from the pickle file.")
        return ""

    logging.info("Removing page numbers from loaded text...")
    cleaned_text = remove_page_numbers(full_text)
    logging.info(f"Text loaded and cleaned (~{len(cleaned_text)} chars).")
    return cleaned_text


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting TTS Script ---")

    # Set the start method for multiprocessing (important for CUDA)
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
             mp.set_start_method("spawn", force=True)
             logging.info("Set multiprocessing start method to 'spawn'.")
    except Exception as e:
         logging.warning(f"Could not set multiprocessing start method: {e}")


    # --- 1. Load and Prepare Text ---
    # The load_text function needs to be adapted to handle the list format from post_processor.py
    # For now, assuming load_text is modified or replaced by logic to extract text from the list structure
    # Example:
    # try:
    #     with open(INPUT_PICKLE_FILE, "rb") as f:
    #         processed_data_list = pickle.load(f)
    #     # Extract 'processed_result' from each successful entry
    #     text_segments = [entry['processed_result'] for entry in processed_data_list if entry and entry.get('status') == 'success' and entry.get('processed_result')]
    #     text = "\n\n".join(text_segments) # Join segments with double newline
    #     if not text:
    #          logging.critical("No text extracted from processed data list. Exiting.")
    #          exit(1)
    # except Exception as e:
    #      logging.critical(f"Failed to load or process text from {INPUT_PICKLE_FILE}: {e}. Exiting.")
    #      exit(1)
    # --- Using original load_text for now, needs update ---
    text = load_text(INPUT_PICKLE_FILE) # TODO: Update this logic
    if text is None: # Check if loading failed
        logging.critical(f"Failed to load text from {INPUT_PICKLE_FILE}. Exiting.")
        exit(1)
    if not text: # Check if text is empty
        logging.critical("Loaded text is empty. Nothing to synthesize. Exiting.")
        exit(1)

    # Split the text into paragraphs/chunks suitable for TTS
    # Using the imported split_text from processing_audio
    paragraphs = split_text(text)
    if not paragraphs:
        logging.critical("Text splitting resulted in no paragraphs. Exiting.")
        exit(1)
    logging.info(f"Split text into {len(paragraphs)} paragraphs.")

    # --- 2. Prepare for Parallel Processing ---
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logging.warning("No CUDA GPUs detected by PyTorch. TTS will likely run on CPU.")
        # Adapt logic if CPU-only processing is desired, or exit
        # For now, we'll proceed assuming TTS library handles CPU fallback
        num_processes = 1 # Use a single process for CPU
    else:
        num_processes = num_gpus
        logging.info(f"Found {num_gpus} CUDA GPUs available.")

    # Split paragraphs into chunks for each process
    if len(paragraphs) < num_processes:
        num_processes = len(paragraphs) # Don't start more processes than paragraphs
        logging.info(f"Reduced number of processes to {num_processes} (number of paragraphs).")

    chunk_size = (len(paragraphs) + num_processes - 1) // num_processes # Ceiling division
    chunks = [
        paragraphs[i : i + chunk_size] for i in range(0, len(paragraphs), chunk_size)
    ]
    logging.info(f"Divided paragraphs into {len(chunks)} chunks for parallel processing.")

    # --- 3. Initialize TTS Model (Once) ---
    tts_model = None
    try:
        logging.info("Initializing TTS model (xtts_v2)...")
        # Initialize TTS - progress_bar might not work well with multiprocessing logs
        tts_model = TTS(model_name="xtts_v2", progress_bar=False, gpu=(num_gpus > 0))
        logging.info("TTS model initialized successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialize TTS model: {e}. Exiting.")
        exit(1)


    # --- 4. Run TTS in Parallel ---
    process_args = []
    for i, chunk in enumerate(chunks):
        gpu_id = i % num_gpus if num_gpus > 0 else -1 # Assign GPU ID or -1 for CPU
        # Pass the speaker wav path to process_text
        process_args.append((tts_model, chunk, gpu_id, f"{OUTPUT_FILE_PREFIX}_proc{i}", SPEAKER_WAV_PATH))

    logging.info(f"Starting TTS processing using {num_processes} processes...")
    try:
        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(process_text, process_args)
        logging.info("Parallel TTS processing finished.")
    except Exception as e:
        logging.error(f"Error during multiprocessing TTS: {e}")
        # Consider cleanup or partial results handling here

    # --- 5. Collect and Concatenate Audio Files ---
    # Use glob to find all generated parts, more robust than calculating indices
    search_pattern = f"{OUTPUT_FILE_PREFIX}_proc*_part_*.wav"
    logging.info(f"Searching for generated audio files matching: {search_pattern}")
    audio_files = sorted(glob.glob(search_pattern)) # Sort to maintain order

    if not audio_files:
        logging.error("No audio part files were found after processing. Check logs.")
    else:
        logging.info(f"Found {len(audio_files)} audio part files.")
        # Concatenate audio files
        concatenation_successful = concatenate_audio_files(audio_files, FINAL_OUTPUT_WAV)

        # --- 6. Clean up ---
        if concatenation_successful:
            logging.info("Cleaning up individual audio part files...")
            files_removed = 0
            files_failed = 0
            for file_path in audio_files:
                try:
                    os.remove(file_path)
                    files_removed += 1
                except OSError as e:
                    files_failed += 1
                    logging.warning(f"Failed to remove temporary file {file_path}: {e}")
            logging.info(f"Cleanup complete. Removed {files_removed} files, failed to remove {files_failed}.")
        else:
             logging.warning("Concatenation failed. Skipping cleanup of part files.")


    logging.info("--- TTS Script Finished ---")
