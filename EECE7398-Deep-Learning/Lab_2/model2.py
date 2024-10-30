import re
import html
from underthesea import sent_tokenize, text_normalize, word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize as en_sent_tokenize
from nltk.tokenize import RegexpTokenizer

import re
import html
from underthesea import sent_tokenize, text_normalize, word_tokenize
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
    timestamp_pattern = r'\d+\s+(?:\d{2}[:]\d{2}[:]\d{2},\d{3})\s*-->\s*(?:\d{2}[:]\d{2}[:]\d{2},\d{3})'
    text = re.sub(timestamp_pattern, '', text)

    # Step 7: Remove extra spaces and clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 8: Replace '$' with 'đô la'
    text = re.sub(r'\$', ' đô la', text)
    text = re.sub(r'\bUSD\b', 'đô la', text)
    text = re.sub(r'\bdollars\b', 'đô la', text, flags=re.IGNORECASE)

    # Step 9: Remove commas within numbers
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # Step 10: Convert numerals using context-aware function
    text = convert_vietnamese_number_words_to_digits(text)
    print(text)

    # Step 11: Fix spacing around punctuation
    text = fix_punctuation_spacing(text)

    # Step 12: Sentence segmentation
    sentences = sent_tokenize(text)

    # Step 13: Normalize each sentence
    normalized_sentences = [text_normalize(sentence) for sentence in sentences]

    # Step 14: Word segmentation using underthesea
    tokenized_sentences = [word_tokenize(sentence) for sentence in normalized_sentences]

    return tokenized_sentences


def convert_vietnamese_number_words_to_digits(text):
    import re
    from underthesea import pos_tag
    from vietnam_number import w2n

    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    processed_text = ''

    # Set of Vietnamese number words
    number_words = {
        'không', 'một', 'mốt', 'hai', 'ba', 'bốn', 'tư', 'năm', 'lăm',
        'sáu', 'bảy', 'tám', 'chín', 'mười', 'mươi', 'trăm',
        'nghìn', 'ngàn', 'triệu', 'tỷ', 'linh', 'lẻ'
    }

    for sentence in sentences:
        pos_tags = pos_tag(sentence)
        new_tokens = []
        i = 0
        while i < len(pos_tags):
            token, tag = pos_tags[i]

            # Split multi-word tokens into individual words
            token_words = token.split()
            lower_token_words = [word.lower() for word in token_words]

            # Handle 'phần trăm' (percent)
            if 'phần' in lower_token_words and 'trăm' in lower_token_words:
                new_tokens.append('%')
                i += 1
                continue

            # Check if the first word is a number word and tagged as 'M'
            if lower_token_words[0] in number_words and tag == 'M':
                numeral_tokens = token_words
                i += 1
                # Collect following tokens that are number words and tagged as 'M'
                while i < len(pos_tags):
                    next_token, next_tag = pos_tags[i]
                    next_token_words = next_token.split()
                    lower_next_token_words = [word.lower() for word in next_token_words]

                    if all(word in number_words for word in lower_next_token_words) and next_tag == 'M':
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
        processed_text += processed_sentence + ' '

    # Clean up extra spaces and return the processed text
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text



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

text_vi = '''Một lần nữa , như đối với Mongolia , Trung Quốc không hề xâm chiếm Nga . Trung Quốc chỉ cho thuê nước Nga .
Tôi gọi điều này là hiện tượng toàn cầu hoá theo kiểu Trung Hoa .
Bây giờ đây có thể là bản đồ khu vực trong vòng 10 đến 20 năm tới .
Nhưng từ từ đã . Bản đồ này đã 700 năm tuổi .
Đây là bản đồ của Triều Đại Nhà Nguyên , dưới sự lãnh đại của Kubla Khan , cháu nội của Genghis Khan .'''
text_en = 'One hundred twenty-five girls will not be married when they &apos;re 12 years old .'

processed_vi = preprocess_text_vietnamese(text_vi)
processed_en = preprocess_text_english(text_en)

print("Vietnamese Tokens:", processed_vi)
print("English Tokens:", processed_en)
