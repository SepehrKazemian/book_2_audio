import re
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_text(tts_model, paragraphs, gpu_id, file_prefix, speaker_wav="wav_file.wav", language="en"):
    """
    Processes a list of text paragraphs to synthesize speech using a pre-initialized TTS model
    and saves the output to WAV files.

    Args:
        tts_model: An initialized TTS model object (e.g., from TTS.api.TTS).
        paragraphs (list): A list of strings, where each string is a paragraph of text.
        gpu_id (int): The ID of the GPU to use for processing.
        file_prefix (str): The prefix for the output WAV files.
        speaker_wav (str, optional): Path to the speaker WAV file for voice cloning.
                                     Defaults to "wav_file.wav".
        language (str, optional): Language code for the TTS synthesis. Defaults to "en".
    """
    try:
        # Set the GPU device for PyTorch (redundant if TTS model already on GPU, but safe)
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            logging.info(f"Set active CUDA device to GPU {gpu_id}")
        else:
            logging.warning("CUDA not available. TTS might run on CPU.")

        # Process each paragraph
        for i, paragraph in enumerate(paragraphs):
            output_path = f"{file_prefix}_part_{gpu_id}_{i}.wav"
            logging.info(f"Processing paragraph {i} for GPU {gpu_id}. Output: {output_path}")
            try:
                tts_model.tts_to_file(
                    text=paragraph,
                    speaker_wav=speaker_wav,
                    file_path=output_path,
                    language=language,
                    split_sentences=True,  # Keep sentence splitting enabled
                )
                logging.info(f"Successfully saved {output_path}")
            except Exception as e:
                logging.error(f"Error processing paragraph {i} on GPU {gpu_id}: {e}")
                # Optionally, decide whether to continue or stop processing

    except Exception as e:
        logging.error(f"General error in process_text for GPU {gpu_id}: {e}")


def split_text(text):
    """
    Splits a larger text into paragraphs based on double newlines and cleans up
    single newlines within paragraphs.

    Args:
        text (str): The input text to split.

    Returns:
        list: A list of strings, where each string is a processed paragraph.
    """
    # Replace double newlines between lowercase letters with a single space.
    # This helps join paragraphs that might be artificially split by formatting.
    text = re.sub(r"(?<=[a-z])\n\n+(?=[a-z])", " ", text)

    # Split the text into potential paragraphs based on one or more double newlines.
    paragraphs = re.split(r"\n\n+", text)

    processed_paragraphs = []
    for paragraph in paragraphs:
        # Within each paragraph, replace single newlines with spaces,
        # unless the newline is preceded by sentence-ending punctuation (.!?)
        # or followed by an uppercase letter (indicating a potential new sentence start).
        cleaned_paragraph = re.sub(r"(?<![.!?])\n(?![A-Z])", " ", paragraph)
        processed_paragraphs.append(cleaned_paragraph.strip()) # Remove leading/trailing whitespace

    # Filter out any empty strings that might result from splitting
    return [p for p in processed_paragraphs if p]
