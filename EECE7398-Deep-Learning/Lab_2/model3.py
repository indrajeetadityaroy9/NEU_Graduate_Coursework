# ======================
# Import Libraries
# ======================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import MarianTokenizer

import time
import random
from tqdm import tqdm  # For progress bars

# Import necessary modules for mixed precision and learning rate scheduling
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Download necessary NLTK data
nltk.download('punkt')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ======================
# Mount Google Drive
# ======================

from google.colab import drive
drive.mount('/content/drive')

# Adjust the MODEL_PATH to point to a directory in your Google Drive
MODEL_PATH = '/content/drive/My Drive/NMT_Model/'

# Ensure the model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# ======================
# Parameters and Setup
# ======================

# Define separate embedding dimensions
ENCODER_EMBEDDING_SIZE = 128
DECODER_EMBEDDING_SIZE = 512  # Must match HIDDEN_SIZE for MultiheadAttention
HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT = 0.5
NUM_HEADS = 8
EPOCHS = 20
SAVED_MODEL_NAME = 'best_model.pth'

# Initialize the tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Vocabulary sizes
VOCAB_SIZE_ENG = tokenizer.vocab_size
VOCAB_SIZE_DE = tokenizer.vocab_size

print(f'English Vocabulary Size: {VOCAB_SIZE_ENG}')
print(f'German Vocabulary Size: {VOCAB_SIZE_DE}')

# Define padding token ID
PAD_IDX = tokenizer.pad_token_id

# ======================
# Bucketing Parameters
# ======================

# Define length boundaries and batch sizes
boundaries = [8, 16, 32, 64, 128, 256, 512]
batch_sizes = [256, 128, 64, 32, 16, 8, 4, 2]  # Corresponding batch sizes

# ======================
# Data Loading and Preprocessing
# ======================

# Paths to your data files
ENGLISH_DATA_PATH = "train.en"
GERMAN_DATA_PATH = "train.de"

# Function to preprocess German sentences
def preprocess_german_sentence(sentence, tokenizer):
    start_token = tokenizer.bos_token  # Beginning of Sentence
    end_token = tokenizer.eos_token    # End of Sentence
    return f"{start_token} {sentence.strip()} {end_token}"

# ======================
# Dataset and DataLoader
# ======================

class TranslationIterableDataset(IterableDataset):
    def __init__(self, english_path, german_path, tokenizer, boundaries, batch_sizes, shuffle=True):
        self.english_path = english_path
        self.german_path = german_path
        self.tokenizer = tokenizer
        self.boundaries = boundaries
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle

    def __iter__(self):
        return self.data_generator()

    def data_generator(self):
        buckets = {i: [] for i in range(len(self.boundaries) + 1)}
        batch_counts = {i: 0 for i in range(len(self.boundaries) + 1)}
        file_indices = list(range(len(open(self.english_path).readlines())))

        if self.shuffle:
            random.shuffle(file_indices)

        with open(self.english_path, 'r', encoding='utf8') as f_eng, open(self.german_path, 'r', encoding='utf8') as f_de:
            eng_lines = f_eng.readlines()
            de_lines = f_de.readlines()

            # If shuffling is enabled, shuffle the data
            if self.shuffle:
                data = list(zip(eng_lines, de_lines))
                random.shuffle(data)
                eng_lines, de_lines = zip(*data)

            for idx, (eng_sentence, de_sentence) in enumerate(zip(eng_lines, de_lines)):
                eng_sentence = eng_sentence.strip()
                de_sentence = de_sentence.strip()
                if eng_sentence and de_sentence:
                    # Preprocess and tokenize sentences
                    de_sentence = preprocess_german_sentence(de_sentence, self.tokenizer)

                    # Tokenize sentences
                    tokenized_eng = self.tokenizer(
                        eng_sentence,
                        return_tensors='pt',
                        padding=False,
                        truncation=True
                    )
                    tokenized_de = self.tokenizer(
                        de_sentence,
                        return_tensors='pt',
                        padding=False,
                        truncation=True
                    )

                    # Prepare decoder input and output
                    decoder_input_ids = tokenized_de['input_ids'][:, :-1]
                    decoder_output_ids = tokenized_de['input_ids'][:, 1:]

                    data_item = {
                        'encoder_input_ids': tokenized_eng['input_ids'].squeeze(0),
                        'decoder_input_ids': decoder_input_ids.squeeze(0),
                        'decoder_output_ids': decoder_output_ids.squeeze(0)
                    }

                    # Calculate sequence length
                    seq_length = data_item['encoder_input_ids'].size(0) + data_item['decoder_input_ids'].size(0)

                    # Assign to bucket
                    for i, boundary in enumerate(self.boundaries):
                        if seq_length <= boundary:
                            bucket_id = i
                            break
                    else:
                        bucket_id = len(self.boundaries)

                    buckets[bucket_id].append(data_item)
                    batch_counts[bucket_id] += 1

                    # Yield batch if bucket is full
                    batch_size = self.batch_sizes[bucket_id]
                    if len(buckets[bucket_id]) == batch_size:
                        batch = buckets[bucket_id]
                        if self.shuffle:
                            random.shuffle(batch)
                        yield collate_fn(batch)
                        buckets[bucket_id] = []

            # Yield remaining data
            for bucket_id, bucket in buckets.items():
                if bucket:
                    if self.shuffle:
                        random.shuffle(bucket)
                    yield collate_fn(bucket)

# Custom collate function for dynamic padding
def collate_fn(batch):
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    decoder_output_ids = [item['decoder_output_ids'] for item in batch]

    # Pad sequences dynamically
    encoder_input_ids = nn.utils.rnn.pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    decoder_input_ids = nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    decoder_output_ids = nn.utils.rnn.pad_sequence(decoder_output_ids, batch_first=True, padding_value=PAD_IDX)

    return {
        'encoder_input_ids': encoder_input_ids,
        'decoder_input_ids': decoder_input_ids,
        'decoder_output_ids': decoder_output_ids
    }

# Create the dataset
train_dataset = TranslationIterableDataset(
    ENGLISH_DATA_PATH,
    GERMAN_DATA_PATH,
    tokenizer,
    boundaries,
    batch_sizes,
    shuffle=True
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=None,  # Batch size is handled within the dataset
    num_workers=0,    # Set to 0 to avoid multiprocessing issues in Colab
    prefetch_factor=None
)

# For validation, you can create a separate dataset without shuffling
val_dataset = TranslationIterableDataset(
    ENGLISH_DATA_PATH,
    GERMAN_DATA_PATH,
    tokenizer,
    boundaries,
    batch_sizes,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=None,
    num_workers=0,
    prefetch_factor=None
)

# ======================
# Model Architecture
# ======================

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        outputs, (hidden, cell) = self.lstm(embedded)  # outputs: [batch, seq_len, hidden_size]
        return outputs, (hidden, cell)

# Decoder with Multi-Head Attention
class DecoderWithMultiHeadAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=NUM_LAYERS, dropout=DROPOUT, num_heads=NUM_HEADS):
        super(DecoderWithMultiHeadAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs, mask=None):
        # x: [batch_size]
        # hidden: [num_layers, batch, hidden_size]
        # cell: [num_layers, batch, hidden_size]
        # encoder_outputs: [batch, seq_len, hidden_size]

        x = x.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embedding_dim]

        # Prepare for MultiheadAttention (requires [seq_len, batch, embed_dim])
        embedded = embedded.permute(1, 0, 2)  # [1, batch_size, embedding_dim]

        # encoder_outputs: [batch, seq_len, hidden_size] -> [seq_len, batch, hidden_size]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [seq_len, batch, hidden_size]

        # Perform Multihead Attention
        attn_output, attn_weights = self.multihead_attn(embedded, encoder_outputs, encoder_outputs, key_padding_mask=~mask)
        # attn_output: [1, batch, hidden_size]

        attn_output = attn_output.permute(1, 0, 2)  # [batch, 1, hidden_size]

        # Concatenate embedded input and attention output
        rnn_input = torch.cat((embedded.permute(1, 0, 2), attn_output), dim=2)  # [batch, 1, embedding_dim + hidden_size]

        # Pass through LSTM
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))  # output: [batch, 1, hidden_size]

        # Predict the next token
        prediction = self.fc_out(output.squeeze(1))  # [batch, vocab_size]

        return prediction, hidden, cell, attn_weights

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        # src: [batch_size, src_len]
        mask = (src != PAD_IDX)  # [batch_size, src_len]
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, seq_len]
        # trg: [batch_size, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        # Encoder outputs
        encoder_outputs, (hidden, cell) = self.encoder(src)

        # First input to the decoder is the <bos> tokens
        input = trg[:, 0]

        # Create mask
        mask = self.create_mask(src).to(self.device)  # [batch_size, src_len]

        for t in range(1, trg_len):
            # Decode
            output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = output

            # Decide whether to do teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs

# ======================
# Training and Evaluation
# ======================

# Initialize the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize the encoder and decoder with correct embedding dimensions
encoder = Encoder(VOCAB_SIZE_ENG, ENCODER_EMBEDDING_SIZE, HIDDEN_SIZE).to(device)
decoder = DecoderWithMultiHeadAttention(
    VOCAB_SIZE_DE,
    DECODER_EMBEDDING_SIZE,  # Should be 512 to match HIDDEN_SIZE
    HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    num_heads=NUM_HEADS
).to(device)

# Initialize the Seq2Seq model
model = Seq2Seq(encoder, decoder, device).to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # Using PAD_IDX for padding

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Implement learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Initialize GradScaler for mixed precision training
scaler = GradScaler(enabled=(device.type == 'cuda'))

# ======================
# Evaluation Metrics
# ======================

def evaluate_bleu(model, data_loader, tokenizer, device, num_samples=100):
    model.eval()
    smoothing = SmoothingFunction()
    scores = []

    sample_count = 0
    with torch.no_grad():
        for batch in data_loader:
            src = batch['encoder_input_ids'].to(device)
            trg = batch['decoder_output_ids'].to(device)

            for i in range(src.size(0)):
                src_sentence = tokenizer.decode(src[i], skip_special_tokens=True)
                trg_sentence = tokenizer.decode(trg[i], skip_special_tokens=True).split()

                # Translate
                translated = translate(model, src_sentence, tokenizer, tokenizer, device)
                translated_tokens = translated.split()

                # Calculate BLEU score
                score = sentence_bleu([trg_sentence], translated_tokens, smoothing_function=smoothing.method1)
                scores.append(score)

                sample_count += 1
                if sample_count >= num_samples:
                    break
            if sample_count >= num_samples:
                break

    average_bleu = np.mean(scores)
    print(f'Average BLEU score over {num_samples} samples: {average_bleu:.4f}')

# Function to calculate epoch time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Initialize variables for early stopping
best_val_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    model.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Training]", leave=False)
    for batch in train_loader_tqdm:
        src = batch['encoder_input_ids'].to(device)
        trg = batch['decoder_input_ids'].to(device)
        trg_out = batch['decoder_output_ids'].to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            # Forward pass
            output = model(src, trg)  # [batch_size, trg_len, vocab_size]

            # Reshape for loss calculation
            output = output[:, 1:].reshape(-1, output.size(-1))  # [(batch_size * (trg_len - 1)), vocab_size]
            trg_out = trg_out[:, 1:].reshape(-1)  # [(batch_size * (trg_len - 1))]

            # Calculate loss
            loss = criterion(output, trg_out)

        # Backward pass and optimization
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        # Update progress bar
        train_loader_tqdm.set_postfix({'Batch Loss': batch_loss, 'Avg Loss': epoch_loss / (train_loader_tqdm.n + 1)})

    avg_train_loss = epoch_loss / len(train_loader)

    # Validation Phase
    model.eval()
    val_loss = 0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Validation]", leave=False)
    with torch.no_grad():
        for batch in val_loader_tqdm:
            src = batch['encoder_input_ids'].to(device)
            trg = batch['decoder_input_ids'].to(device)
            trg_out = batch['decoder_output_ids'].to(device)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                # Forward pass
                output = model(src, trg, teacher_forcing_ratio=0)  # No teacher forcing during validation

                # Reshape for loss calculation
                output = output[:, 1:].reshape(-1, output.size(-1))
                trg_out = trg_out[:, 1:].reshape(-1)

                # Calculate loss
                loss = criterion(output, trg_out)

            batch_loss = loss.item()
            val_loss += batch_loss

            # Update progress bar
            val_loader_tqdm.set_postfix({'Batch Loss': batch_loss, 'Avg Loss': val_loss / (val_loader_tqdm.n + 1)})

    avg_val_loss = val_loss / len(val_loader)

    # Step the scheduler
    scheduler.step(avg_val_loss)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {avg_train_loss:.4f}')
    print(f'\t Val. Loss: {avg_val_loss:.4f}')

    # Evaluate BLEU Score after each epoch
    evaluate_bleu(model, val_loader, tokenizer, device, num_samples=100)

    # Check for improvement
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, SAVED_MODEL_NAME))
        print(f'\t New best model saved at {os.path.join(MODEL_PATH, SAVED_MODEL_NAME)}')
        trigger_times = 0
    else:
        trigger_times += 1
        print(f'\t No improvement. Trigger times: {trigger_times}/{patience}')
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# ======================
# Inference
# ======================

def translate(model, sentence, tokenizer, tokenizer_de, device, max_len=512):
    model.eval()

    # Tokenize the input sentence
    encoded = tokenizer(sentence, return_tensors='pt', padding=False, truncation=True, max_length=max_len).to(device)
    src = encoded['input_ids']

    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src)

    # Initialize the decoder input with the <bos> token
    decoder_input = torch.tensor([tokenizer_de.bos_token_id], dtype=torch.long).to(device)

    decoded_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            mask = model.create_mask(src)
            output, hidden, cell, attn_weights = model.decoder(decoder_input, hidden, cell, encoder_outputs, mask)
            # Get the highest predicted token
            top1 = output.argmax(1)
            if top1.item() == tokenizer_de.eos_token_id:
                break
            else:
                decoded_tokens.append(tokenizer_de.decode(top1))

            # Next input is current prediction
            decoder_input = top1

    return ' '.join(decoded_tokens)

# ======================
# Saving the Model
# ======================

# Save the final model after training
torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'final_model.pth'))
print(f'Final model saved at {os.path.join(MODEL_PATH, "final_model.pth")}')

# Loading the Model

# Function to load the saved model
def load_model(model_path, encoder, decoder, device):
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
