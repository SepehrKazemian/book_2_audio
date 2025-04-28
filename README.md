# Audiobook Generation Pipeline

This project implements a pipeline to generate an audiobook from a PDF file. It involves several steps, including text extraction, cleaning, LLM-based correction using a fine-tuned model, and text-to-speech conversion.

## Project Structure

```
.
├── pdf_extraction/         # Step 1: Extract text from PDF
│   ├── __init__.py
│   └── text_reader.py
├── text_cleaning/          # Step 2: Initial text cleanup
│   ├── __init__.py
│   ├── line_cleanup.py
│   ├── line_structure.py
│   └── text_cleaning.py
├── llm_finetuning/         # Step 3: Fine-tune LLM for text correction
│   ├── __init__.py
│   ├── data_preparation/   # Scripts to prepare fine-tuning data (e.g., from Wikipedia)
│   │   ├── __init__.py
│   │   ├── dataset_main.py
│   │   ├── dataset_manipulation.py
│   │   ├── dataset_to_text.py
│   │   ├── noise_generator.py
│   │   └── text_to_sample.py
│   └── training/           # Scripts for the fine-tuning process
│       ├── __init__.py
│       ├── fine_tuned_model/ # Directory for saved model adapter (ignored by git)
│       ├── model_loader.py
│       └── spell_correction.py
├── llm_correction/         # Step 3.1: Apply fine-tuned LLM to book text
│   ├── __init__.py
│   ├── result_check.py
│   └── text_llm_correction.py
├── post_llm_processing/    # Step 3.2: Post-processing after LLM correction
│   ├── __init__.py
│   └── post_processor.py
├── tts_conversion/         # Step 4: Convert final text to audio
│   ├── __init__.py
│   ├── processing_audio.py
│   └── tts.py
├── data/                   # Directory for input/output data (ignored by git)
├── utils/                  # Directory for shared utility functions (optional)
│   └── __init__.py
├── unused_code/            # Original notebooks, logs, redundant scripts (ignored by git)
├── book1/                  # Example book data? (ignored by git)
├── .gitattributes
├── .gitignore
└── requirements.txt
```

## Pipeline Overview

1.  **PDF Extraction (`pdf_extraction/`)**:
    *   Reads text content from an input PDF file (`download.pdf` by default).
    *   Uses PyMuPDF (`fitz`) for initial extraction.
    *   Employs Tesseract OCR via `pdf2image` and `pytesseract` as a fallback for pages with potentially garbled text (determined by checking the percentage of valid English words against NLTK and system dictionaries).
    *   Saves the extracted text per page into a pickle file (`download_book_text.pkl` by default, intended for the `data/` directory).

2.  **Text Cleaning (`text_cleaning/`)**:
    *   Loads the extracted text data.
    *   Analyzes line structure (lengths, Roman numerals).
    *   Applies a series of cleanup steps:
        *   Filters unwanted lines (e.g., lines with only digits, single letters).
        *   Concatenates lines belonging to the same paragraph.
        *   Merges isolated bullet points with the following line.
        *   Normalizes whitespace (multiple spaces, tabs) and removes empty lines.
    *   The main script `text_cleaning.py` orchestrates these steps.

3.  **LLM Fine-tuning (`llm_finetuning/`)**:
    *   **Data Preparation (`data_preparation/`)**:
        *   Loads a dataset (e.g., `google/wiki40b` specified in `dataset_main.py`).
        *   Processes the dataset text (e.g., removing sections/articles).
        *   Splits text into samples suitable for the LLM context window.
        *   Introduces artificial noise (character removal, addition, swapping, special characters) to create training pairs (noisy text -> original text).
        *   Formats the data into a prompt structure suitable for fine-tuning (e.g., correcting spelling and identifying sections).
        *   Saves the prepared dataset splits (e.g., `wiki_train_data.pkl`, intended for `data/`).
    *   **Training (`training/`)**:
        *   Loads a base LLM (e.g., `stabilityai/stablelm-zephyr-3b` specified in `model_loader.py`).
        *   Applies 4-bit quantization and prepares the model for k-bit training using PEFT (LoRA).
        *   Loads the prepared training/evaluation data.
        *   Uses `trl.SFTTrainer` to fine-tune the model on the task of correcting the noisy text based on the provided prompt format.
        *   Includes callbacks for early stopping and printing example predictions during training.
        *   Saves the trained PEFT adapter (`fine_tuned_model/` by default).

4.  **LLM Correction (`llm_correction/`)**:
    *   Loads the fine-tuned PEFT model adapter.
    *   Loads the cleaned book text (output from Step 2).
    *   Batches the text according to token limits.
    *   Runs the text batches through the fine-tuned LLM using a specific prompt to perform spell checking and section identification/correction.
    *   Optionally uses multiple model instances for parallel processing via `concurrent.futures`.
    *   Compares the LLM output with the input batch using fuzzy matching and cosine similarity (`result_check.py`).
    *   Saves the results (original batch, processed result, comparison scores) periodically and finally to a pickle file (`text_correction_results.pkl` by default, intended for `data/`).

5.  **Post-LLM Processing (`post_llm_processing/`)**:
    *   Loads the results from the LLM correction step.
    *   Removes specific entries based on their index/key (e.g., pages 0, 2, 3 corresponding to cover, ToC, etc., as defined in `post_processor.py`).
    *   Filters out any entries that failed during the LLM correction step.
    *   Saves the final filtered/processed text data (`final_processed_book.pkl` by default, intended for `data/`).

6.  **TTS Conversion (`tts_conversion/`)**:
    *   Loads the final processed text data (likely needs modification to load from Step 5's output).
    *   Concatenates the text content.
    *   Removes residual page number patterns.
    *   Splits the text into manageable paragraphs/chunks.
    *   Initializes a TTS model (`coqui/TTS xtts_v2` specified in `tts.py`).
    *   Uses multiprocessing (`mp.Pool`) to distribute TTS generation across available GPUs (or CPU).
    *   Generates audio segments for each text chunk using a specified speaker WAV (`wav_file.wav` by default).
    *   Concatenates the generated audio segments using `pydub`, applying short fades.
    *   Exports the final audiobook (`final_output.wav` by default, intended for `data/`).
    *   Cleans up intermediate audio part files.

## Setup and Running

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Ensure tesseract OCR is installed and configured if needed by pdf_extraction
    # Ensure necessary NLTK data is downloaded (handled partially in text_reader.py)
    ```
2.  **Configuration:**
    *   Review constants defined at the beginning of each script, especially file paths (input PDF, output directories, data files, model paths) and model IDs. Adjust as necessary or consider using a dedicated configuration file/environment variables.
    *   Ensure the input PDF (e.g., `download.pdf`) and speaker WAV file (e.g., `wav_file.wav`) are placed correctly (likely in the `data/` directory or specified path).
3.  **Running the Pipeline:**
    *   Execute the scripts sequentially, ensuring the output of one step is correctly used as input for the next.
    *   Example (run from the project root directory):
        ```bash
        python pdf_extraction/text_reader.py
        python text_cleaning/text_cleaning.py
        # Fine-tuning steps (may require significant resources)
        python llm_finetuning/data_preparation/dataset_main.py
        python llm_finetuning/training/spell_correction.py
        # Correction and Post-processing
        python llm_correction/text_llm_correction.py
        python post_llm_processing/post_processor.py
        # TTS Conversion
        python tts_conversion/tts.py
        ```
    *   Note: Input/output paths within the scripts might need adjustment based on where data files are stored (e.g., changing `../` paths to `data/`).

## Notes

*   The project uses multiprocessing extensively. Ensure your system has sufficient resources (CPU cores, RAM, GPU memory if using GPUs).
*   Error handling has been added, but complex pipelines can fail in various ways. Check logs for details.
*   Model IDs, dataset names, and specific parameters (batch sizes, learning rates, etc.) are hardcoded as constants; consider externalizing these into configuration files for flexibility.
*   The `.gitignore` file is configured to exclude common data files, model checkpoints, logs, and cache files. Ensure any sensitive information (API keys, specific book content files, etc.) is added to `.gitignore` and not committed.
*   Consider using environment variables or a dedicated configuration file (e.g., using `python-dotenv` or `hydra`) for managing paths, model IDs, and other parameters instead of hardcoding them as constants in the scripts.
