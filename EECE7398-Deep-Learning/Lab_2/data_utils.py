import re
import html
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch

# Load PhoBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("huynguyen208/marian-finetuned-kde4-en-to-vi-190322")


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

    # Step 13: Fix spacing around punctuation
    text = fix_punctuation_spacing(text)

    return text


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


def preprocess_and_tokenize(text):
    # Preprocess Vietnamese text
    cleaned_text = preprocess_text_vietnamese(text)
    # Tokenize
    tokens = tokenizer.tokenize(cleaned_text)

    # Merge subwords into complete words
    clean_tokens = []
    current_word = ""
    for token in tokens:
        if token.startswith('▁'):
            if current_word:
                clean_tokens.append(current_word)  # Append the current word
            current_word = token[1:]  # Start a new word
        else:
            current_word += token  # Add subword to the current word

    # Append the last word if any
    if current_word:
        clean_tokens.append(current_word)

    # Separate punctuation
    final_tokens = []
    for token in clean_tokens:
        # Separate punctuation at the end of each word if present
        split_tokens = re.findall(r'\w+|[.,!?;]', token)
        final_tokens.extend(split_tokens)

    return final_tokens


def replace_large_number_words_vietnamese(text):
    large_numbers = {
        'nghìn tỷ': '1000000000000',
        'tỷ': '1000000000',
        'triệu': '1000000',
        'nghìn': '1000',
    }

    # Replace specific large number phrases to avoid spacing issues
    for word, replacement in large_numbers.items():
        pattern = rf'(\d+(\.\d+)?\s*{word})'
        text = re.sub(pattern,
                      lambda match: str(int(float(match.group(1).replace(word, "").strip()) * int(replacement))), text)

    return text


# Test with example
text = "Đó là một câu hỏi đáng giá cả triệu dollar , phải không ?"
clean_tokens = preprocess_and_tokenize(text)
print("Clean Tokens:", clean_tokens)
