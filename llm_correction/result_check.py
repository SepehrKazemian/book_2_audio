import re
import logging
import torch # Add missing import
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

# --- Constants ---
# Markers typically inserted during processing that should be removed/replaced for comparison
MARKERS_TO_REPLACE = ["_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_", "_NEWLINE_"]
# Threshold for fuzzy word matching (adjust as needed)
FUZZY_MATCH_THRESHOLD = 90

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_text_for_comparison(text: str, markers_to_remove: List[str] = MARKERS_TO_REPLACE) -> str:
    """
    Preprocesses text for comparison by removing specified markers,
    normalizing certain patterns, and cleaning whitespace.

    Args:
        text (str): The input text string.
        markers_to_remove (List[str]): A list of marker strings to replace with spaces.

    Returns:
        str: The preprocessed text string.
    """
    if not isinstance(text, str):
         logging.warning("Input to preprocess_text_for_comparison is not a string. Returning as is.")
         return text # Or raise error

    processed_text = text
    # Replace specified markers (case-insensitive) with spaces
    for marker in markers_to_remove:
        # Using regex for case-insensitive replacement might be better if needed:
        # pattern = re.compile(re.escape(marker), re.IGNORECASE)
        # processed_text = pattern.sub(" ", processed_text)
        # Simple case-sensitive replacement based on original code:
        processed_text = processed_text.replace(marker, " ") # Assuming markers are exact case

    # Normalize hyphenated line breaks (e.g., "word- \nword" -> "word-word") - check if needed
    # processed_text = processed_text.replace("- \n", "-") # Requires multi-line handling if text contains newlines
    # Original code's replacement (might be specific to data format):
    processed_text = processed_text.replace("- ", "-")

    # Normalize ellipsis
    processed_text = processed_text.replace("…", "...")

    # Add space after period if followed directly by an uppercase letter (potential sentence join)
    processed_text = re.sub(r"(?<=\.)([A-Z])", r" \1", processed_text)

    # Collapse multiple spaces into one
    processed_text = re.sub(r"\s+", " ", processed_text).strip()

    return processed_text


def calculate_fuzzy_word_match_score(original_text: str, processed_text: str) -> Dict[str, Any]:
    """
    Calculates a score based on consecutive fuzzy word matches between two sentences.

    Compares words sequentially and stops at the first mismatch below the threshold.

    Args:
        original_text (str): The original text string.
        processed_text (str): The processed/generated text string to compare against the original.

    Returns:
        Dict[str, Any]: A dictionary containing:
            'match_count': Number of consecutively matched words.
            'matched_chars': Total characters in the matched words from the original text.
            'last_match_ratio': Fuzzy ratio of the last compared word pair (or 0 if no words).
    """
    results = {'match_count': 0, 'matched_chars': 0, 'last_match_ratio': 0}
    try:
        words_original = original_text.split()
        words_processed = processed_text.split()

        if not words_original or not words_processed:
            logging.debug("One or both texts have no words for fuzzy matching.")
            return results

        match_count = 0
        matched_chars = 0
        last_ratio = 0

        # Iterate through words in the processed text, comparing with original
        for i, p_word in enumerate(words_processed):
            if i < len(words_original):
                o_word = words_original[i]
                try:
                    match_ratio = fuzz.ratio(o_word.lower(), p_word.lower())
                    last_ratio = match_ratio # Store last calculated ratio
                except Exception as e:
                     logging.warning(f"Fuzzywuzzy error comparing '{o_word}' and '{p_word}': {e}")
                     match_ratio = 0 # Treat error as mismatch

                if match_ratio > FUZZY_MATCH_THRESHOLD:
                    match_count += 1
                    matched_chars += len(o_word)
                else:
                    # Mismatch found, stop comparison
                    logging.debug(f"Fuzzy match failed at word {i}: '{o_word}' vs '{p_word}' (Ratio: {match_ratio})")
                    break
            else:
                # Processed text is longer than original, stop comparison
                logging.debug(f"Processed text longer than original at word {i}. Stopping fuzzy match.")
                break

        results['match_count'] = match_count
        results['matched_chars'] = matched_chars
        results['last_match_ratio'] = last_ratio

    except Exception as e:
        logging.error(f"Error during fuzzy word matching: {e}", exc_info=True)
        # Return default results on error

    return results


def calculate_cosine_similarity(
    text1: str, text2: str, tokenizer: Any, model: Any
) -> Optional[float]:
    """
    Calculates the cosine similarity between the embeddings of two texts.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.
        tokenizer (Any): The tokenizer object.
        model (Any): The model object capable of generating embeddings/logits.

    Returns:
        Optional[float]: The cosine similarity score (0.0 to 1.0), or None if calculation fails.
    """
    if not text1 or not text2:
         logging.warning("One or both texts are empty for cosine similarity calculation.")
         return None

    try:
        logging.debug("Generating embeddings for cosine similarity...")
        # Generate embeddings (assuming model outputs logits, take mean as simple embedding)
        # This might need adjustment based on the specific model architecture
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512) # Limit length
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Move inputs to the same device as the model
        device = model.device
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}

        with torch.no_grad(): # Ensure no gradients are calculated
             outputs1 = model(**inputs1)
             outputs2 = model(**inputs2)

        # Use mean of last hidden state as embedding (common practice)
        # Check model output structure - might be outputs1.last_hidden_state
        if hasattr(outputs1, 'last_hidden_state') and hasattr(outputs2, 'last_hidden_state'):
             embedding1 = outputs1.last_hidden_state.mean(dim=1)
             embedding2 = outputs2.last_hidden_state.mean(dim=1)
        # Fallback to logits if last_hidden_state is not available (less ideal)
        elif hasattr(outputs1, 'logits') and hasattr(outputs2, 'logits'):
             logging.warning("Using mean of logits as embedding for cosine similarity.")
             embedding1 = outputs1.logits.mean(dim=1)
             embedding2 = outputs2.logits.mean(dim=1)
        else:
             logging.error("Model output does not contain 'last_hidden_state' or 'logits'. Cannot calculate embeddings.")
             return None


        # Calculate cosine similarity
        # Ensure embeddings are on CPU and converted to numpy
        similarity_matrix = cosine_similarity(
            embedding1.cpu().detach().numpy(),
            embedding2.cpu().detach().numpy()
        )
        similarity_score = similarity_matrix[0][0]

        logging.info(f"Cosine Similarity Score: {similarity_score:.4f}")
        return float(similarity_score) # Ensure float type

    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}", exc_info=True)
        return None


def compare_text_results(
    original_text: str,
    processed_result: str,
    tokenizer: Any,
    model: Any
) -> Dict[str, Any]:
    """
    Compares an original text with a processed result using fuzzy matching and cosine similarity.

    Args:
        original_text (str): The original text.
        processed_result (str): The processed text (e.g., from LLM).
        tokenizer (Any): The tokenizer for embedding generation.
        model (Any): The model for embedding generation.

    Returns:
        Dict[str, Any]: A dictionary containing comparison metrics:
                        'fuzzy_match_score': Results from calculate_fuzzy_word_match_score.
                        'cosine_similarity': Result from calculate_cosine_similarity (or None).
    """
    logging.debug("Starting text comparison...")

    # Preprocess both texts for fairer comparison
    # Original text might not need marker removal, but whitespace cleaning is good
    text_clean = preprocess_text_for_comparison(original_text, markers_to_remove=[])
    result_clean = preprocess_text_for_comparison(processed_result, markers_to_remove=MARKERS_TO_REPLACE)
    logging.debug(f"Cleaned Original Text (start): {text_clean[:100]}...")
    logging.debug(f"Cleaned Processed Result (start): {result_clean[:100]}...")


    # Calculate Fuzzy Match Score
    fuzzy_score = calculate_fuzzy_word_match_score(text_clean, result_clean)
    logging.info(f"Fuzzy Match Score: {fuzzy_score}")

    # Calculate Cosine Similarity
    cosine_sim = calculate_cosine_similarity(text_clean, result_clean, tokenizer, model)
    # Logging is done inside the function

    comparison_results = {
        'fuzzy_match_score': fuzzy_score,
        'cosine_similarity': cosine_sim
    }
    logging.debug(f"Comparison Results: {comparison_results}")

    return comparison_results

# Example of how this module might be called (e.g., from text_llm_correction.py)
# if __name__ == "__main__":
#     # This is just a placeholder example
#     # You would need to load actual model, tokenizer, and text data
#     logging.basicConfig(level=logging.DEBUG) # Set level to DEBUG for detailed logs
#     class MockTokenizer:
#         def __call__(self, text, **kwargs): return {'input_ids': torch.randint(0, 1000, (1, len(text.split())))}
#         def encode(self, text, **kwargs): return [random.randint(0,1000) for _ in text] # Dummy encode
#     class MockModel:
#         device = 'cpu'
#         def __call__(self, **kwargs):
#             input_ids = kwargs.get('input_ids', torch.tensor([[1]]))
#             # Return dummy logits or last_hidden_state
#             return type('obj', (object,), {'last_hidden_state': torch.randn(1, input_ids.shape[1], 10)})()
#
#     mock_tokenizer = MockTokenizer()
#     mock_model = MockModel()
#
#     original = "This is the original sentence with _START_SECTION_ marker."
#     processed = "This is the original sentence with marker ." # Example processed output
#
#     results = compare_text_results(original, processed, mock_tokenizer, mock_model)
#     print("\n--- Example Comparison ---")
#     print(f"Original:  '{original}'")
#     print(f"Processed: '{processed}'")
#     print(f"Results:   {results}")
