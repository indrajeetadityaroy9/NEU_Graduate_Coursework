import re

def find_encoded_elements(file_paths):
    # Regular expression patterns to match various encoded elements

    # Match numeric character references like &#91;
    numeric_entity_pattern = r'&#\d+;'

    # Match named character references like &apos;, &quot;, &amp;, &lt;, &gt;
    named_entity_pattern = r'&[a-zA-Z]+;'

    # Match possible double-encoded entities like &amp; lt ; or &amp; amp ; amp ;
    double_encoded_pattern = r'(&[a-zA-Z]+;(\s?[a-zA-Z]+)*\s?;)'

    # Combine all patterns
    combined_pattern = f'({numeric_entity_pattern})|({named_entity_pattern})|({double_encoded_pattern})'

    # Use a set to store unique entities
    entities_found = set()

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Find all matches in the line
                matches = re.findall(combined_pattern, line)
                for match in matches:
                    # Each match is a tuple; filter out empty strings
                    entity = ''.join(filter(None, match))
                    # Clean up any extra whitespace
                    entity = ' '.join(entity.split())
                    entities_found.add(entity)

    # Output all unique entities found
    print("\nEntities found:")
    for entity in sorted(entities_found):
        print(entity)

# Example usage
file_paths = ['train.en', 'train.vi']
find_encoded_elements(file_paths)
