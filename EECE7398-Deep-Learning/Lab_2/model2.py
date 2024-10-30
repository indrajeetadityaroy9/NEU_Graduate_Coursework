import re
import html
from underthesea import sent_tokenize, text_normalize, word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize as en_sent_tokenize
from nltk.tokenize import RegexpTokenizer
from pyvi import ViTokenizer, ViPosTagger
from nltk.tokenize import sent_tokenize as en_sent_tokenize
from nltk.tokenize import word_tokenize as en_word_tokenize
from word2number import w2n as en_w2n
from vietnam_number import w2n


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

    # Step 14: Sentence segmentation using underthesea
    sentences = sent_tokenize(text)

    # Step 15: Normalize each sentence
    normalized_sentences = [text_normalize(sentence) for sentence in sentences]

    tokenized_sentences = [ViTokenizer.tokenize(sentence) for sentence in normalized_sentences]

    # Step 17: Split tokens into lists
    tokenized_sentences = [sentence.split() for sentence in tokenized_sentences]

    # **Step 18: Normalize tokens by removing underscores before subword tokenization**
    tokenized_sentences = [
        [token.replace('_', ' ') for token in sentence] for sentence in tokenized_sentences
    ]

    return tokenized_sentences


def replace_large_number_words_vietnamese(text):
    # Define large number words and their numeric equivalents
    large_numbers = {
        'nghìn tỷ': '1000000000000',  # trillion
        'tỷ': '1000000000',  # billion
        'triệu': '1000000',  # million
        'nghìn': '1000',  # thousand
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

    # Words that indicate 'năm' means 'year' if they follow it
    year_indicators = {'trước', 'sau', 'nay', 'đó', 'tới'}

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

            # Check if the first word is a number word
            if lower_token_words[0] in number_words:
                # Check if the number is part of an identifier
                if is_part_of_identifier(i):
                    # Do not convert; keep as is
                    new_tokens.extend(token_words)
                    i += 1
                    continue

                # Special handling for 'năm'
                if lower_token_words[0] == 'năm':
                    # Check if 'năm' means 'year'
                    if i + 1 < len(tokens_list):
                        next_token = tokens_list[i + 1].lower()
                        if next_token in year_indicators:
                            new_tokens.extend(token_words)
                            i += 1
                            continue

                numeral_tokens = token_words
                i += 1
                # Collect following tokens that are number words
                while i < len(tokens_list):
                    next_token = tokens_list[i]
                    next_tag = tags_list[i]
                    next_token_words = next_token.replace('_', ' ').split()
                    lower_next_token_words = [word.lower() for word in next_token_words]

                    if all(word in number_words for word in lower_next_token_words):
                        numeral_tokens.extend(next_token_words)
                        i += 1
                    else:
                        break

                numeral_phrase = ' '.join(numeral_tokens).lower()
                try:
                    number = w2n(numeral_phrase)
                    new_tokens.append(str(number))
                except ValueError:
                    # If conversion fails, keep the original tokens
                    new_tokens.extend(numeral_tokens)
            else:
                new_tokens.extend(token_words)
                i += 1

        processed_sentence = ' '.join(new_tokens)
        processed_sentence = processed_sentence.replace('_', ' ')
        processed_text += processed_sentence + ' '

    # Clean up extra spaces and return the processed text
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


def preprocess_text_english(text):
    # Step 1: Fix double-encoded entities
    text = re.sub(r'(&amp;\s?lt\s?;)+', '<', text, flags=re.IGNORECASE)
    text = re.sub(r'(&amp;\s?gt\s?;)+', '>', text, flags=re.IGNORECASE)
    text = re.sub(r'(&amp;\s?amp\s?;)+', '&', text, flags=re.IGNORECASE)

    # Step 2: Replace any remaining single-encoded entities
    text = re.sub(r'&lt;', '<', text, flags=re.IGNORECASE)
    text = re.sub(r'&gt;', '>', text, flags=re.IGNORECASE)
    text = re.sub(r'&quot;', '"', text, flags=re.IGNORECASE)
    text = re.sub(r'&apos;', "'", text, flags=re.IGNORECASE)
    text = re.sub(r'&amp;', '&', text, flags=re.IGNORECASE)
    text = re.sub(r'[“”]', '"', text)

    # Step 3: Decode any remaining HTML entities
    text = html.unescape(text)

    # Step 4: Remove brackets if desired
    text = text.replace('[', '').replace(']', '')

    # Step 5: Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Step 6: Remove extra spaces and clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 7: Remove commas within numbers
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # Step 8: Replace '$' with 'dollars'
    text = re.sub(r'\$\s*([\d]+(?:[\d.,]*\d)?)', r'\1 dollars', text)

    # Step 9: Replace 'dollar(s)' with correct pluralization
    def replace_dollar_plural(match):
        number_str = match.group(1)
        number = float(number_str.replace(',', ''))
        return f"{number_str} dollar" if number == 1 else f"{number_str} dollars"

    text = re.sub(r'\b([\d]+(?:[\d.,]*\d)?)\s*dollars?\b', replace_dollar_plural, text, flags=re.IGNORECASE)

    # Updated Step 10: Replace standalone large number words with their numeric equivalents
    text = replace_large_number_words(text)

    # Step 11: Convert number words to digits
    text = convert_number_words_to_digits(text)

    # Step 12: Sentence segmentation
    sentences = en_sent_tokenize(text)

    # Step 13: Word tokenization
    tokenized_sentences = [en_word_tokenize(sentence) for sentence in sentences]

    # Step 14: Replace backticks with standard quotes
    tokenized_sentences = [
        [token.replace('``', '"').replace("''", '"') for token in sentence]
        for sentence in tokenized_sentences
    ]

    return tokenized_sentences


def replace_large_number_words(text):
    import nltk
    tokens = nltk.word_tokenize(text)
    large_numbers = {
        'million': '1000000',
        'billion': '1000000000',
        'trillion': '1000000000000'
    }
    number_words_set = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                        'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
                        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
                        'thousand'}
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_lower = token.lower()
        if token_lower in ('million', 'millions', 'billion', 'billions', 'trillion', 'trillions'):
            # Check if previous token is not a number word
            if i == 0 or tokens[i - 1].lower() not in number_words_set:
                # Replace with numeric value
                singular = token_lower.rstrip('s')
                number = large_numbers[singular]
                new_tokens.append(number)
                i += 1
            else:
                new_tokens.append(token)
                i += 1
        else:
            new_tokens.append(token)
            i += 1
    return ' '.join(new_tokens)


def convert_number_words_to_digits(text):
    import nltk
    tokens = nltk.word_tokenize(text)
    number_words_set = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                        'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
                        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
                        'thousand', 'million', 'millions', 'billion', 'billions', 'trillion', 'trillions', 'and'}
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if is_number_word(token, number_words_set):
            number_tokens = []
            while i < len(tokens) and is_number_word(tokens[i], number_words_set):
                number_tokens.append(tokens[i])
                i += 1
            number_str = ' '.join(number_tokens)
            # Replace hyphens with spaces and remove 'and'
            number_str_cleaned = number_str.replace('-', ' ').replace(' and ', ' ').lower()
            # Handle plural forms by converting to singular
            number_str_cleaned = number_str_cleaned.replace('millions', 'million')
            number_str_cleaned = number_str_cleaned.replace('billions', 'billion')
            number_str_cleaned = number_str_cleaned.replace('trillions', 'trillion')
            try:
                number = en_w2n.word_to_num(number_str_cleaned)
                new_tokens.append(str(number))
            except ValueError:
                # If conversion fails, return the original tokens
                new_tokens.extend(number_tokens)
        else:
            new_tokens.append(token)
            i += 1
    return ' '.join(new_tokens)


def is_number_word(token, number_words_set):
    token_lower = token.lower()
    if token_lower in number_words_set:
        return True
    elif '-' in token_lower:
        parts = token_lower.split('-')
        return all(part in number_words_set for part in parts)
    else:
        return False


text_vi = 'Ba trăm năm mươi ngàn người mỗi ngày bước qua Quảng Trường Thời Đại , và mọi người đã cố gắng trong nhiều năm để tạo sự thay đổi .'
text_en = 'And when they finished , we asked them , &quot; Do you want to build another one ? &quot; for $ 2.40 , $ 2.10 , and so on , until at some point people said , &quot; No more . It &apos;s not worth it for me . &quot;'

processed_vi = preprocess_text_vietnamese(text_vi)
processed_en = preprocess_text_english(text_en)

print("Vietnamese Tokens:", processed_vi)
print("English Tokens:", processed_en)

sentences = [' '.join(tokens) for tokens in processed_vi]
text_for_subword = '\n'.join(sentences)
print(text_for_subword)