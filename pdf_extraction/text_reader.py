fimport nltk
from nltk.corpus import wordnet
import re
import fitz  # PyMuPDF
from tqdm import tqdm
import pytesseract
from pdf2image import convert_from_path
import os
import pickle
import logging
from typing import Set, Dict, List, Any, Optional

# --- Constants ---
LINUX_DICT_PATH = "/usr/share/dict/words" # System path, keep as is
# Note: TESSDATA_PREFIX might be better set globally or via environment config
TESSDATA_PREFIX_PATH = r"/usr/share/tesseract-ocr/4.00/tessdata/" # System path, keep as is
# Note: TESSDATA config string might be environment-specific
TESSDATA_CONFIG = r'--tessdata-dir "{}" --psm 4 --oem 1'.format(TESSDATA_PREFIX_PATH) # System path, keep as is
# !! IMPORTANT: Consider making paths configurable !!
DEFAULT_PDF_PATH = "data/download.pdf" # Assuming input PDF is in data/
DEFAULT_OUTPUT_PICKLE_PATH = "data/extracted_book_text.pkl" # Output to data/
GARBLED_TEXT_THRESHOLD = 0.8  # Min percentage of valid words to not trigger OCR
MIN_TEXT_LENGTH_FOR_CHECK = 10 # Min characters on page to check for garbled text

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Setup (Consider externalizing) ---
try:
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX_PATH
    logging.info(f"Set TESSDATA_PREFIX to: {TESSDATA_PREFIX_PATH}")
except Exception as e:
    logging.warning(f"Could not set TESSDATA_PREFIX environment variable: {e}")

# --- NLTK Data Download ---
def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('corpora/words')
        logging.debug("NLTK 'words' corpus found.")
    except LookupError:
        logging.info("NLTK 'words' corpus not found. Downloading...")
        nltk.download('words', quiet=True)
        logging.info("Downloaded NLTK 'words'.")
    try:
        nltk.data.find('corpora/wordnet')
        logging.debug("NLTK 'wordnet' corpus found.")
    except LookupError:
        logging.info("NLTK 'wordnet' corpus not found. Downloading...")
        nltk.download('wordnet', quiet=True)
        logging.info("Downloaded NLTK 'wordnet'.")

# --- Word Dictionary Loading ---
def load_dictionary_words() -> Optional[Set[str]]:
    """
    Loads English words from NLTK (words, wordnet) and Linux dictionary.

    Returns:
        Optional[Set[str]]: A set of unique English words, or None if loading fails.
    """
    all_words = set()
    try:
        # Ensure NLTK data is available
        download_nltk_data()

        # NLTK words corpus
        nltk_valid_words = set(nltk.corpus.words.words())
        logging.info(f"Loaded {len(nltk_valid_words)} words from NLTK 'words' corpus.")
        all_words.update(w.lower() for w in nltk_valid_words if len(w) >= 2)

        # NLTK WordNet
        wordnet_words = set(word.name().split(".")[0].replace("_", "") for word in wordnet.all_synsets())
        logging.info(f"Loaded {len(wordnet_words)} words from NLTK WordNet.")
        all_words.update(w.lower() for w in wordnet_words if len(w) >= 2)

        # Linux dictionary
        if os.path.exists(LINUX_DICT_PATH):
            try:
                with open(LINUX_DICT_PATH, "r") as f:
                    linux_words_raw = f.read()
                # Simple split, filter length >= 2, and lowercase
                linux_dict_words = set(
                    word.lower()
                    for word in re.findall(r"\b[a-zA-Z-]{2,}\b", linux_words_raw) # Find words with letters/hyphens
                )
                logging.info(f"Loaded {len(linux_dict_words)} words from {LINUX_DICT_PATH}.")
                all_words.update(linux_dict_words)
            except IOError as e:
                logging.warning(f"Could not read Linux dictionary at {LINUX_DICT_PATH}: {e}")
        else:
            logging.warning(f"Linux dictionary not found at {LINUX_DICT_PATH}.")

        # Remove potential empty strings if any slipped through
        all_words.discard("")

        logging.info(f"Total unique words loaded: {len(all_words)}")
        return all_words

    except Exception as e:
        logging.error(f"Failed to load dictionary words: {e}")
        return None

# --- Text Analysis ---
def is_text_garbled(text: str, valid_words: Set[str]) -> float:
    """
    Calculates the percentage of valid English words in a given text.

    Args:
        text (str): The text to analyze.
        valid_words (Set[str]): A set of known valid English words (lowercase).

    Returns:
        float: The percentage (0.0 to 1.0) of valid words found in the text.
               Returns 0.0 if no suitable tokens are found.
    """
    # Find potential words (alphabetic, at least 2 chars long)
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text)
    if not tokens:
        return 0.0  # No words found

    valid_word_count = sum(1 for token in tokens if token.lower() in valid_words)
    valid_word_percentage = valid_word_count / len(tokens)
    logging.debug(f"Garbled check: {valid_word_count}/{len(tokens)} valid words ({valid_word_percentage:.2%})")
    return valid_word_percentage

# --- OCR Function ---
def ocr_page(pdf_path: str, page_number: int) -> Optional[str]:
    """
    Performs OCR on a specific page of a PDF.

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): The 0-based index of the page to OCR.

    Returns:
        Optional[str]: The extracted text from OCR, or None if an error occurs.
    """
    logging.info(f"Performing OCR on page {page_number} of {pdf_path}...")
    try:
        # pdf2image uses 1-based page numbers
        images = convert_from_path(
            pdf_path, first_page=page_number + 1, last_page=page_number + 1, dpi=300 # Increase DPI for better OCR
        )
        if not images:
            logging.warning(f"pdf2image returned no image for page {page_number}.")
            return None

        text = pytesseract.image_to_string(
            images[0], lang="eng", config=TESSDATA_CONFIG
        )
        logging.info(f"OCR successful for page {page_number}. Extracted ~{len(text)} chars.")
        return text
    except ImportError:
         logging.error("pdf2image or pytesseract not installed correctly.")
         return None
    except Exception as e:
        logging.error(f"Error during OCR for page {page_number}: {e}")
        return None

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path: str, valid_words: Set[str]) -> Optional[Dict[int, List[Any]]]:
    """
    Extracts text from each page of a PDF, using OCR as a fallback for garbled text.

    Args:
        pdf_path (str): Path to the PDF file.
        valid_words (Set[str]): Set of valid English words for garbled text check.

    Returns:
        Optional[Dict[int, List[Any]]]: A dictionary where keys are 0-based page numbers
                                        and values are lists [extracted_text, 0].
                                        Returns None if the PDF cannot be opened.
    """
    pages_data: Dict[int, List[Any]] = {}
    document: Optional[fitz.Document] = None

    try:
        logging.info(f"Opening PDF: {pdf_path}")
        document = fitz.open(pdf_path)
        num_pages = len(document)
        logging.info(f"PDF has {num_pages} pages.")

        for page_number in tqdm(range(num_pages), desc="Extracting text from PDF pages"):
            extracted_text = ""
            needs_ocr = False
            try:
                page = document.load_page(page_number)
                text = page.get_text("text")

                if len(text.strip()) > MIN_TEXT_LENGTH_FOR_CHECK:
                    valid_perc = is_text_garbled(text, valid_words)
                    if valid_perc < GARBLED_TEXT_THRESHOLD:
                        logging.warning(f"Page {page_number}: Text potentially garbled ({valid_perc:.2%}). Triggering OCR.")
                        needs_ocr = True
                    else:
                        extracted_text = text
                        logging.debug(f"Page {page_number}: Extracted text via fitz.")
                else:
                    logging.info(f"Page {page_number}: Text length too short ({len(text.strip())} chars). Triggering OCR.")
                    needs_ocr = True

                if needs_ocr:
                    ocr_text = ocr_page(pdf_path, page_number)
                    extracted_text = ocr_text if ocr_text is not None else "" # Use OCR text or empty string

            except Exception as e:
                logging.error(f"Error processing page {page_number}: {e}. Attempting OCR as fallback.")
                ocr_text = ocr_page(pdf_path, page_number)
                extracted_text = ocr_text if ocr_text is not None else ""

            pages_data[page_number] = [extracted_text, 0] # Store text and placeholder 0

        return pages_data

    except fitz.fitz.FileNotFoundError:
        logging.error(f"PDF file not found at: {pdf_path}")
        return None
    except Exception as e:
        logging.error(f"Failed to open or process PDF {pdf_path}: {e}")
        return None
    finally:
        if document:
            document.close()
            logging.info(f"Closed PDF: {pdf_path}")


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting PDF Text Extraction Script ---")

    # Load dictionary
    all_words = load_dictionary_words()
    if all_words is None:
        logging.critical("Could not load dictionary words. Exiting.")
        exit(1)

    # Extract text from PDF
    pdf_pages_content = extract_text_from_pdf(DEFAULT_PDF_PATH, all_words)

    if pdf_pages_content is None:
        logging.critical(f"Failed to extract text from PDF: {DEFAULT_PDF_PATH}. Exiting.")
        exit(1)

    # Save extracted text
    try:
        logging.info(f"Saving extracted text to {DEFAULT_OUTPUT_PICKLE_PATH}...")
        with open(DEFAULT_OUTPUT_PICKLE_PATH, "wb") as file:
            pickle.dump(pdf_pages_content, file)
        logging.info("Successfully saved extracted text.")
    except IOError as e:
        logging.error(f"Failed to save extracted text to {DEFAULT_OUTPUT_PICKLE_PATH}: {e}")
    except pickle.PicklingError as e:
        logging.error(f"Failed to pickle extracted text data: {e}")

    logging.info("--- PDF Text Extraction Script Finished ---")
