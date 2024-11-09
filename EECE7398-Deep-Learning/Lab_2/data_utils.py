import re
import html
from pyvi import ViTokenizer, ViPosTagger
from vietnam_number import w2n  # Ensure this package is installed: pip install vietnam_number

def preprocess_text_vietnamese(text):
    # Step 1: Fix double-encoded entities
    text = re.sub(r'(&amp;\s?lt\s?;)+', '<', text, flags=re.IGNORECASE)
    text = re.sub(r'(&amp;\s?gt\s?;)+', '>', text, flags=re.IGNORECASE)
    text = re.sub(r'(&amp;\s?amp\s?;)+', '&', text, flags=re.IGNORECASE)
    text = re.sub(r'(&amp;\s?apos\s?;)+', "'", text, flags=re.IGNORECASE)
    text = re.sub(r'(&amp;\s?quot\s?;)+', '"', text, flags=re.IGNORECASE)

    # Step 2: Replace any remaining single-encoded entities
    text = re.sub(r'&lt;', '<', text, flags=re.IGNORECASE)
    text = re.sub(r'&gt;', '>', text, flags=re.IGNORECASE)
    text = re.sub(r'&amp;', '&', text, flags=re.IGNORECASE)
    text = re.sub(r'&apos;', "'", text, flags=re.IGNORECASE)
    text = re.sub(r'&quot;', '"', text, flags=re.IGNORECASE)

    # Step 3: Decode numeric character references
    text = html.unescape(text)

    # Step 4: Remove brackets if desired
    text = text.replace('[', '').replace(']', '')

    # Step 5: Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Step 6: Remove timestamps
    timestamp_pattern = r'\d+\s+(?:\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?:\d{2}:\d{2}:\d{2},\d{3})'
    text = re.sub(timestamp_pattern, '', text)

    # Step 7: Remove extra spaces and clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 8: Replace '$' with 'đô la'
    text = re.sub(r'\$', ' đô la', text)
    text = re.sub(r'\bUSD\b', 'đô la', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdollars?\b', 'đô la', text, flags=re.IGNORECASE)

    # Step 9: Remove commas within numbers
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # Step 10: Remove dots used as thousands separators
    text = remove_dots_in_numbers(text)

    # Step 11: Replace standalone large number words with their numeric equivalents
    text = replace_large_number_words_vietnamese(text)

    # Step 12: Convert Vietnamese number words to digits
    text = convert_vietnamese_number_words_to_digits(text)

    # Step 13: Fix spacing around punctuation
    text = fix_punctuation_spacing(text)

    return text


def replace_large_number_words_vietnamese(text):
    # Define large number words and their numeric equivalents
    large_numbers = {
        'nghìn tỷ': '1000000000000',  # trillion
        'tỷ': '1000000000',           # billion
        'triệu': '1000000',            # million
        'nghìn': '1000',               # thousand
    }

    # Sort the keys by length in descending order to match longer phrases first
    sorted_large_numbers = sorted(large_numbers.keys(), key=len, reverse=True)

    # Function to replace standalone large number words
    def replace_large_numbers(match):
        word = match.group(0)
        return large_numbers.get(word.lower(), word)

    # Replace standalone large number words
    for word in sorted_large_numbers:
        # Use word boundaries to match standalone words
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, replace_large_numbers, text, flags=re.IGNORECASE)

    return text


def convert_vietnamese_number_words_to_digits(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    processed_text = ''

    # Set of Vietnamese number words
    number_words = {
        'không', 'một', 'mốt', 'hai', 'ba', 'bốn', 'tư', 'năm', 'lăm',
        'sáu', 'bảy', 'tám', 'chín', 'mười', 'mươi', 'trăm',
        'nghìn', 'ngàn', 'triệu', 'tỷ', 'linh', 'lẻ', 'phần'
    }

    # Words that indicate 'năm' means 'year' if they follow or precede it
    year_indicators = {'nhiều', 'mỗi', 'trong', 'vào', 'với', 'trước', 'sau', 'nay', 'đó', 'tới'}

    # Keywords that may precede identifiers
    identifier_keywords = {'chuyến bay', 'số hiệu', 'mã', 'biển số', 'số', 'kí hiệu', 'địa chỉ'}

    for sentence in sentences:
        # Tokenize and POS tag the sentence using Pyvi
        tokens = ViTokenizer.tokenize(sentence)
        tokens_list, tags_list = ViPosTagger.postagging(tokens)

        new_tokens = []
        i = 0
        while i < len(tokens_list):
            token = tokens_list[i]
            tag = tags_list[i]

            # Split multi-word tokens into individual words
            token_words = token.replace('_', ' ').split()
            lower_token_words = [word.lower() for word in token_words]

            # Function to check if the current token is part of an identifier
            def is_part_of_identifier(index):
                # Check previous tokens for identifier keywords
                for offset in range(1, 3):  # Look back up to 2 tokens
                    if index - offset >= 0:
                        prev_token = tokens_list[index - offset].lower().replace('_', ' ')
                        if prev_token in identifier_keywords:
                            return True
                return False

            # **Step A: Handle "năm" as "year" first**
            if lower_token_words[0] == "năm" and i > 0 and tokens_list[i - 1].lower() in year_indicators:
                # "năm" is "year", do not convert
                new_tokens.extend(token_words)
                i += 1
                continue

            # **Step B: Process numeral phrases**
            if lower_token_words[0] in number_words:
                # Check if the number is part of an identifier
                if is_part_of_identifier(i):
                    # Do not convert; keep as is
                    new_tokens.extend(token_words)
                    i += 1
                    continue

                # Collect numeral tokens
                numeral_tokens = token_words.copy()
                i += 1
                # Collect following tokens that are number words
                while i < len(tokens_list):
                    next_token = tokens_list[i]
                    next_token_words = next_token.replace('_', ' ').split()
                    lower_next_token_words = [word.lower() for word in next_token_words]

                    if all(word in number_words for word in lower_next_token_words):
                        numeral_tokens.extend(next_token_words)
                        i += 1
                    else:
                        break

                # Normalize "ngàn" to "nghìn" for consistency
                numeral_phrase = ' '.join(numeral_tokens).lower().replace('ngàn', 'nghìn')

                try:
                    # Convert using w2n
                    number = w2n(numeral_phrase)
                    new_tokens.append(str(number))
                except ValueError:
                    # If conversion fails, keep the original tokens
                    # Uncomment the next line for debugging
                    # print(f"Conversion failed for phrase: '{numeral_phrase}'")
                    new_tokens.extend(numeral_tokens)
            else:
                # Handle "năm" in context of "year" (already handled above)
                # Keep other tokens as is
                new_tokens.extend(token_words)
                i += 1

        # Reconstruct the sentence
        processed_sentence = ' '.join(new_tokens)
        processed_sentence = processed_sentence.replace('_', ' ')
        processed_text += processed_sentence + ' '

    # Clean up extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text


def remove_dots_in_numbers(text):
    def process_number(match):
        number = match.group()
        if '.' in number:
            parts = number.split('.')
            # The first part can be any length; other parts should be exactly 3 digits
            if all(len(part) == 3 for part in parts[1:]) and len(parts[-1]) == 3:
                # Likely thousands separators; remove dots
                return ''.join(parts)
        return number  # Return the number unchanged if it's not matching the pattern

    # Pattern to find numbers with dots
    pattern = r'\b\d+(?:\.\d+)+\b'
    return re.sub(pattern, process_number, text)


def fix_punctuation_spacing(text):
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:"\'’“”])', r'\1', text)
    # Ensure space after punctuation
    text = re.sub(r'([.,!?;:"\'’“”])(\w)', r'\1 \2', text)
    return text




# Test with example
text = "nhưng để đạt tới con số 500 triệu người thì còn khó khăn hơn rất rất nhiều"
processed_text = preprocess_text_vietnamese(text)
print("Processed Text:", processed_text)
