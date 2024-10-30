import re
import html
from underthesea import sent_tokenize, text_normalize, word_tokenize
from nltk.tokenize import sent_tokenize as en_sent_tokenize
from nltk.tokenize import RegexpTokenizer

import re
import html
from underthesea import sent_tokenize, text_normalize, word_tokenize


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
    timestamp_pattern = r'\d+\s+(?:\d{2}\s*[:]\s*\d{2}\s*[:]\s*\d{2},\d{3})\s*-->\s*(?:\d{2}\s*[:]\s*\d{2}\s*[:]\s*\d{2},\d{3})'
    text = re.sub(timestamp_pattern, '', text)

    # Step 7: Remove extra spaces and clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 8: Replace '$' with 'đô la'
    text = re.sub(r'\$', ' đô la', text)

    text = re.sub(r'\bUSD\b', 'đô la', text)

    # Step 9: Replace 'dollars' with 'đô la'
    text = re.sub(r'\bdollars\b', 'đô la', text, flags=re.IGNORECASE)

    # **New Step 10: Remove commas within numbers**
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # Step 11: Fix spacing around punctuation
    text = fix_punctuation_spacing(text)

    # Step 12: Sentence segmentation
    sentences = sent_tokenize(text)

    # Step 13: Normalize each sentence
    normalized_sentences = [text_normalize(sentence) for sentence in sentences]

    # Step 14: Word segmentation using underthesea
    tokenized_sentences = [word_tokenize(sentence) for sentence in normalized_sentences]

    return tokenized_sentences


def fix_punctuation_spacing(text):
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:"\'’“”])', r'\1', text)
    # Ensure space after punctuation if not present
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

    # **New Step 7: Remove commas within numbers**
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # **Updated Step 8: Replace '$' with 'dollars'**
    # Handle cases like '$5,000', '$ 5,000', etc.
    text = re.sub(r'\$\s*([\d]+(?:[\d.,]*\d)?)', r'\1 dollars', text)

    # **Updated Step 9: Replace 'dollar(s)' with correct pluralization**
    def replace_dollar_plural(match):
        number_str = match.group(1)
        number = float(number_str.replace(',', ''))
        return f"{number_str} dollar" if number == 1 else f"{number_str} dollars"

    text = re.sub(r'\b([\d]+(?:[\d.,]*\d)?)\s*dollars?\b', replace_dollar_plural, text, flags=re.IGNORECASE)

    # Step 10: Sentence segmentation
    sentences = en_sent_tokenize(text)

    # Step 11: Word tokenization
    from nltk.tokenize import word_tokenize as en_word_tokenize
    tokenized_sentences = [en_word_tokenize(sentence) for sentence in sentences]

    # Step 12: Replace backticks with standard quotes
    tokenized_sentences = [
        [token.replace('``', '"').replace("''", '"') for token in sentence]
        for sentence in tokenized_sentences
    ]

    return tokenized_sentences


# Example usage
text_vi = 'Và ta có ở đây -- Mọi người có thấy 25 ô tròn màu tím bên trái bạn , và 25 , coi như là ô màu vàng bên phải ?'
text_en = 'One hundred twenty-five girls will not be married when they &apos;re 12 years old .'

processed_vi = preprocess_text_vietnamese(text_vi)
processed_en = preprocess_text_english(text_en)

print("Vietnamese Tokens:", processed_vi)
print("English Tokens:", processed_en)
