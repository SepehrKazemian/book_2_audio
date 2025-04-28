import re
import logging
from typing import List, Tuple

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def filter_unwanted_lines(check_line: List[str], lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filters lines based on predefined patterns, but only considers lines
    marked as 'check' in the corresponding check_line list.

    Args:
        check_line (List[str]): A list indicating whether the corresponding line in `lines`
                                should be checked ('check') or ignored. Length must match `lines`.
        lines (List[str]): The list of text lines to filter.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the filtered list of lines
                                     and the correspondingly filtered check_line list.
                                     Returns original lists if lengths don't match.
    """
    if len(check_line) != len(lines):
        logging.error("filter_unwanted_lines: Mismatch between check_line and lines length. Returning original lists.")
        return lines, check_line

    # Pre-compile regex patterns for efficiency
    unwanted_patterns = [
        # re.compile(r"^\\x0"),  # Example: Lines starting with form feed
        re.compile(r"^\s*\d+\s*$"),  # Lines containing only digits
        re.compile(r"^\s*[A-Za-z]\s*$"),  # Lines containing only a single letter
        re.compile(r"^\s*_+\s*$"),  # Lines containing only underscores/whitespace
    ]
    logging.debug(f"Filtering {len(lines)} lines based on {len(unwanted_patterns)} patterns.")

    delete_indices = []
    for idx in range(len(check_line)):
        # Only process lines marked for checking
        if check_line[idx] == "check":
            line_content = lines[idx]
            # Check if the line matches any of the unwanted patterns
            if any(pattern.match(line_content) for pattern in unwanted_patterns):
                logging.debug(f"Marking line {idx} for deletion (unwanted pattern): '{line_content[:50]}...'")
                delete_indices.append(idx)

    # Delete marked lines in reverse order to maintain correct indices
    if delete_indices:
        logging.info(f"Deleting {len(delete_indices)} unwanted lines.")
        for idx in sorted(delete_indices, reverse=True):
            try:
                del lines[idx]
                del check_line[idx]
            except IndexError:
                 logging.warning(f"Index {idx} out of bounds during deletion, list may have changed unexpectedly.")
                 # This shouldn't happen with the current logic but added as safety
                 pass # Continue if index somehow becomes invalid

    return lines, check_line


def paragraph_line_concat(check_line: List[str], lines: List[str]) -> List[str]:
    """
    Concatenates lines that are likely part of the same paragraph.

    It iterates backwards, joining a line with the previous one if the current line
    doesn't start with a bullet/list marker and the *previous* line is not marked 'check'.

    Args:
        check_line (List[str]): A list indicating status of lines. Used to prevent
                                concatenation if the preceding line is marked 'check'.
        lines (List[str]): The list of text lines.

    Returns:
        List[str]: The list of lines with paragraphs potentially concatenated.
                   Returns original list if lengths don't match.
    """
    if len(check_line) != len(lines):
        logging.error("paragraph_line_concat: Mismatch between check_line and lines length. Returning original list.")
        return lines

    # Comprehensive regex pattern to match common list/bullet point starts
    bullet_point_pattern = re.compile(
        r"^\s*("
        r"[\d•◦∙·‣⁃-]"  # Digits and various bullet characters
        r"|[a-zA-Z]\)"  # Letter followed by parenthesis (e.g., a) )
        r"|[a-zA-Z]\."  # Letter followed by period (e.g., a.)
        # Roman numerals (simplified, might need refinement for complex cases)
        r"|(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?"
        r")\s+" # Require some space after the marker
    )
    logging.debug("Starting paragraph line concatenation.")

    # Store indices to delete *after* modifications to avoid index issues during iteration
    delete_indices = []
    # Iterate backwards from the second-to-last line down to the second line (index 1)
    for idx in range(len(lines) - 1, 0, -1): # Stop at index 1 (second line)
        current_line = lines[idx]
        previous_line_idx = idx - 1

        # Skip if the previous line is marked 'check' (meaning it might be a header or special line)
        if check_line[previous_line_idx] == "check":
            logging.debug(f"Skipping concat at line {idx}: Previous line {previous_line_idx} is 'check'.")
            continue

        # Check if the current line starts with a bullet/list marker
        if not bullet_point_pattern.match(current_line.strip()):
            # If not a list item, assume it's part of the previous line's paragraph
            logging.debug(f"Concatenating line {idx} onto line {previous_line_idx}.")
            lines[previous_line_idx] += " " + current_line.strip() # Modify previous line
            delete_indices.append(idx) # Mark current line for deletion

    # Delete marked lines in reverse order
    if delete_indices:
        logging.info(f"Deleting {len(delete_indices)} lines after concatenation.")
        for idx in sorted(delete_indices, reverse=True):
             try:
                 del lines[idx]
                 # We don't need to delete from check_line here as it's not returned
             except IndexError:
                 logging.warning(f"Index {idx} out of bounds during deletion in paragraph_line_concat.")
                 pass

    return lines


def bullet_point_cleanup(lines: List[str]) -> List[str]:
    """
    Cleans up isolated bullet points by merging them with the following line.

    Identifies lines containing only a bullet/list marker and merges them
    with the content of the next line.

    Args:
        lines (List[str]): The list of text lines.

    Returns:
        List[str]: The list of lines with isolated bullet points cleaned up.
    """
    # Pattern to match lines containing *only* a bullet/digit marker and whitespace
    isolated_bullet_pattern = re.compile(r"^\s*[\d•◦∙·‣⁃-]\s*$")
    logging.debug("Starting isolated bullet point cleanup.")

    modified_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if the current line is an isolated bullet point
        if isolated_bullet_pattern.match(line.strip()):
            # If it is, and there's a next line, merge them
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                merged_line = line.strip() + " " + next_line.strip()
                modified_lines.append(merged_line)
                logging.debug(f"Merged isolated bullet line {i} with line {i+1}.")
                i += 2  # Skip the next line since it's been merged
            else:
                # If it's the last line, keep it as is (though unlikely useful)
                modified_lines.append(line)
                logging.warning(f"Isolated bullet point found on the last line ({i}). Kept as is.")
                i += 1
        else:
            # If not an isolated bullet, just add the line
            modified_lines.append(line)
            i += 1

    return modified_lines


def paragraph_line_cleanup(lines: List[str]) -> List[str]:
    """
    Performs basic cleanup within each line:
    - Replaces multiple spaces with a single space.
    - Removes tab characters.
    - Removes lines containing only whitespace.

    Args:
        lines (List[str]): The list of text lines.

    Returns:
        List[str]: The list of cleaned lines.
    """
    processed_lines = []
    logging.debug("Starting final paragraph line cleanup (whitespace, tabs, empty lines).")
    for i, line in enumerate(lines):
        # Replace multiple spaces with a single space
        cleaned_line = re.sub(r" +", " ", line)
        # Remove tab characters entirely
        cleaned_line = re.sub(r"\t+", "", cleaned_line)
        # Keep the line only if it contains non-whitespace characters after stripping
        if cleaned_line.strip():
            processed_lines.append(cleaned_line)
        else:
             logging.debug(f"Removing empty/whitespace-only line at original index {i}.")

    return processed_lines
