import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ---------------------- Hyperparameters ----------------------
batch_size_saved = 32  # Batch size used when saving tokenized batches
epochs = 4
learning_rate = 1e-3
accumulation_steps = 4  # Gradient accumulation steps
embedding_size = 128
hidden_size = 512
max_len = 100  # Max sequence length
num_words = 10000
vocab_size = num_words + 1  # +1 for 'unk' token

# ---------------------- Device Configuration ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- Data Loading and Tokenization ----------------------
data_path = "/Users/indrajeetadityaroy/Desktop/EECE7398/Lab_2/data/eng2de/"  # Adjust this path

# Ensure the data path exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data path {data_path} does not exist.")

# Load English and German sentences
with open(os.path.join(data_path, "train.en"), encoding="utf8") as f:
    english = f.readlines()
with open(os.path.join(data_path, "train.de"), encoding="utf8") as f:
    german = ["starttt " + line.strip() + " enddd" for line in f.readlines()]

print(f"Total English sentences: {len(english)}")
print(f"Total German sentences: {len(german)}")

# Initialize Tokenizers with oov_token
tokenizer_eng = Tokenizer(num_words=num_words, oov_token='unk')
tokenizer_eng.fit_on_texts(english)
tokenizer_ger = Tokenizer(num_words=num_words, oov_token='unk')
tokenizer_ger.fit_on_texts(german)


# Function to pre-tokenize and save data in batches
def save_tokenized_batches(tokenizer_eng, tokenizer_ger, english, german, batch_size=32):
    english_data, german_data = [], []
    for i, (eng_line, ger_line) in enumerate(zip(english, german)):
        # Tokenize
        english_tokens = tokenizer_eng.texts_to_sequences([eng_line.strip()])[0]
        german_tokens = tokenizer_ger.texts_to_sequences([ger_line.strip()])[0]

        # Replace tokens >= num_words with oov_token index (1)
        english_tokens = [token if token < num_words else 1 for token in english_tokens]
        german_tokens = [token if token < num_words else 1 for token in german_tokens]

        # Pad sequences
        english_padded = pad_sequences([english_tokens], maxlen=max_len, padding='post', truncating='post', value=0)[0]
        german_padded = pad_sequences([german_tokens], maxlen=max_len, padding='post', truncating='post', value=0)[0]

        # Append to batch
        english_data.append(english_padded)
        german_data.append(german_padded)

        # Save in batches
        if (i + 1) % batch_size == 0:
            np.save(os.path.join(data_path, f"english_batch_{i // batch_size}.npy"), np.array(english_data))
            np.save(os.path.join(data_path, f"german_batch_{i // batch_size}.npy"), np.array(german_data))
            english_data, german_data = [], []  # Reset for next batch

            if (i + 1) % (batch_size * 1000) == 0:
                print(f"Saved {i + 1} / {len(english)} sentences into batches.")

    # Save any remaining data
    if english_data:
        np.save(os.path.join(data_path, f"english_batch_{(i // batch_size) + 1}.npy"), np.array(english_data))
        np.save(os.path.join(data_path, f"german_batch_{(i // batch_size) + 1}.npy"), np.array(german_data))
    print("Pre-tokenization and batch saving completed.")


# Check if batches already exist to avoid redundant processing
existing_batches = len([name for name in os.listdir(data_path) if "english_batch_" in name])
expected_batches = len(english) // batch_size_saved
if existing_batches < expected_batches:
    print("Starting pre-tokenization and batch saving...")
    save_tokenized_batches(tokenizer_eng, tokenizer_ger, english, german, batch_size=batch_size_saved)
else:
    print("Tokenized batches already exist. Skipping pre-tokenization.")


# ---------------------- Dataset and DataLoader ----------------------
class TranslationDataset(Dataset):
    def __init__(self, data_path, num_batches):
        self.data_path = data_path
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        english_batch = np.load(os.path.join(self.data_path, f"english_batch_{idx}.npy"))
        german_batch = np.load(os.path.join(self.data_path, f"german_batch_{idx}.npy"))

        # Each batch is [batch_size_saved, max_len]
        return torch.tensor(english_batch, dtype=torch.long), \
            torch.tensor(german_batch[:, :-1], dtype=torch.long), \
            torch.tensor(german_batch[:, 1:], dtype=torch.long)


# Initialize dataset and dataloader
num_batches = len([name for name in os.listdir(data_path) if "english_batch_" in name])
train_dataset = TranslationDataset(data_path, num_batches)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # DataLoader batch_size=1

print(f"Number of batches: {num_batches}")
print(f"DataLoader batch size: {train_loader.batch_size}")


# ---------------------- Model Definitions ----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=3, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch_size_saved, max_len, embedding_size]
        print(f"Encoder input shape before LSTM: {x.shape}")  # Debugging shape check
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        x = self.embedding(x)  # Shape: [batch_size_saved, max_len -1, embedding_size]
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        outputs = self.fc(outputs)  # Shape: [batch_size_saved, max_len -1, vocab_size]
        return outputs, hidden, cell


# Initialize models, optimizer, and scaler for mixed precision
encoder = Encoder(vocab_size, embedding_size, hidden_size).to(device)
decoder = Decoder(vocab_size, embedding_size, hidden_size).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)
scaler = GradScaler()


# ---------------------- Training Function ----------------------
def train():
    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for i, (encoder_input, decoder_input, decoder_output) in enumerate(progress_bar):
            # Move tensors to device
            encoder_input = encoder_input.to(device)  # Shape: [1, batch_size_saved, max_len]
            decoder_input = decoder_input.to(device)  # Shape: [1, batch_size_saved, max_len - 1]
            decoder_output = decoder_output.to(device)  # Shape: [1, batch_size_saved, max_len - 1]

            # Remove the DataLoader's batch dimension
            encoder_input = encoder_input.squeeze(0)  # Shape: [batch_size_saved, max_len]
            decoder_input = decoder_input.squeeze(0)  # Shape: [batch_size_saved, max_len - 1]
            decoder_output = decoder_output.squeeze(0)  # Shape: [batch_size_saved, max_len - 1]

            # Debugging batch shape
            print(f"Batch size: {encoder_input.size(0)}, Sequence length: {encoder_input.size(1)}")

            # Forward pass with mixed precision
            with autocast(enabled=device.type == 'cuda'):
                hidden, cell = encoder(encoder_input)
                outputs, _, _ = decoder(decoder_input, hidden, cell)
                loss = criterion(outputs.view(-1, vocab_size), decoder_output.view(-1)) / accumulation_steps

            # Backward pass and gradient accumulation
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(encoder_optimizer)
                scaler.step(decoder_optimizer)
                scaler.update()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)

            # To prevent excessive logging, you can comment out the following line after verification
            if i == 0 and epoch == 0:
                print(f"Encoder input shape before LSTM: {encoder_input.shape}")

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")


# ---------------------- Translate Function ----------------------
def translate(sentence):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Tokenize and pad the input sentence
        tokens = tokenizer_eng.texts_to_sequences([sentence])[0]
        encoder_input = pad_sequences([tokens], maxlen=max_len, padding='post', truncating='post', value=0)
        encoder_input = torch.tensor(encoder_input, dtype=torch.long).to(device)  # Shape: [1, max_len]

        # Get encoder outputs
        hidden, cell = encoder(encoder_input)

        # Initialize decoder input with the start token
        start_token = tokenizer_ger.word_index.get('starttt', 1)  # Default to 'unk' if not found
        decoder_input = torch.tensor([[start_token]], dtype=torch.long).to(device)  # Shape: [1, 1]

        translated_sentence = []

        for _ in range(50):  # Limit translation to 50 tokens
            with autocast(enabled=device.type == 'cuda'):
                outputs, hidden, cell = decoder(decoder_input, hidden, cell)
                # Get the highest probability token
                token = outputs.argmax(2)  # Shape: [1,1]
                word = tokenizer_ger.index_word.get(token.item(), 'unk')
                if word == 'enddd':
                    break
                translated_sentence.append(word)
                # Update decoder input for the next step
                decoder_input = token

    return ' '.join(translated_sentence)


# ---------------------- Calculate BLEU Score ----------------------
def calculate_bleu(reference, candidate):
    reference = reference.lower().split()
    candidate = candidate.lower().split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference], candidate, smoothing_function=smoothing)


# ---------------------- Main Execution ----------------------
if __name__ == '__main__':
    train()

    # Test translation and BLEU score
    test_sentence = "This is a test sentence."
    translated_sentence = translate(test_sentence)
    print("\n--- Translation Test ---")
    print("Original:", test_sentence)
    print("Translated:", translated_sentence)

    reference_sentence = "Dies ist ein Testsatz."
    bleu_score = calculate_bleu(reference_sentence, translated_sentence)
    print(f"BLEU Score: {bleu_score:.2f}")


