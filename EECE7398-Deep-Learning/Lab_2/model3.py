from datasets import load_dataset

# Load the WMT14 dataset for English-German
dataset = load_dataset("wmt14", "de-en")

# Access the train split
train_data = dataset['train']

# Open the files in write mode
with open("train.en", "w", encoding="utf-8") as f_en, open("train.de", "w", encoding="utf-8") as f_de:
    for i, example in enumerate(train_data):
        english_sentence = example['translation']['en']
        german_sentence = example['translation']['de']

        # Write English and German sentences to their respective files
        f_en.write(english_sentence + "\n")
        f_de.write(german_sentence + "\n")

        # Optional alignment check
        if i < 5:  # Check and print the first 5 pairs for confirmation
            print(f"Pair {i + 1}:")
            print("English:", english_sentence)
            print("German:", german_sentence)
            print("----------")

print("Files 'train.en' and 'train.de' created with aligned English and German sentences.")
