# utils/text_utils.py
import re

def trim_sentences(text: str, max_sentences: int) -> str:
    if not text or max_sentences <= 0:
        return ""

    # Replace actual newline characters with spaces to help sentence splitting
    # Corrected line:
    processed_text = text.replace("\n", " ")

    # Split by common sentence terminators, keeping the terminator with the sentence.
    sentences = re.split(r'(?<=[.!?])\s*|(?<=[.!?][\'"])\s*', processed_text)

    # Filter out empty strings that might result from multiple spaces or newlines
    valid_sentences = [s.strip() for s in sentences if s and s.strip()]

    if not valid_sentences:
        return ""

    # Take up to max_sentences
    trimmed_list = valid_sentences[:max_sentences]
    result = " ".join(trimmed_list)

    # Simplified punctuation addition: if the result is not empty
    # and doesn't end with standard punctuation, add a period.
    # This is a common approach, though it might not be perfect for all edge cases.
    if result and not re.search(r"[.!?]$", result):
        # Only add punctuation if we actually truncated the sentences OR
        # if the last sentence taken (even if it was the only one) didn't have punctuation.
        if len(valid_sentences) > max_sentences or \
           (len(trimmed_list) > 0 and not re.search(r"[.!?]$", trimmed_list[-1])):
            if not result.endswith(('.', '!', '?')): # Double check just in case
                result += '.'

    return result

if __name__ == '__main__':
    # Test cases
    test_text_1 = "This is sentence one. This is sentence two! And sentence three? Sentence four just ends"
    test_text_2 = "A single sentence without punctuation"
    test_text_3 = "First. Second."
    test_text_4 = "One long sentence that does not end with punctuation at all"
    test_text_5 = "Hello world."
    test_text_6 = "Sentence A.\nSentence B. Sentence C" # Test with newline

    print(f"'{trim_sentences(test_text_1, 2)}'")
    print(f"'{trim_sentences(test_text_1, 1)}'")
    print(f"'{trim_sentences(test_text_1, 4)}'")
    print(f"'{trim_sentences(test_text_2, 1)}'")
    print(f"'{trim_sentences(test_text_3, 3)}'")
    print(f"'{trim_sentences(test_text_4, 1)}'")
    print(f"'{trim_sentences(test_text_5, 1)}'")
    print(f"'{trim_sentences(test_text_6, 1)}'")
    print(f"'{trim_sentences(test_text_6, 2)}'")
    print(f"'{trim_sentences(test_text_6, 3)}'")
    print(f"'{trim_sentences('', 2)}'")
