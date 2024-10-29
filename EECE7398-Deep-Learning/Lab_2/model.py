import triton
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import MarianTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler  # Updated import for latest PyTorch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load the MarianMT tokenizer and dataset
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
dataset = load_dataset("wmt14", "de-en")

# Define tokenization function
def tokenize_translation(batch):
    src_texts = [item['en'] for item in batch['translation']]
    tgt_texts = [item['de'] for item in batch['translation']]
    inputs = tokenizer(src_texts, text_target=tgt_texts, padding=True, truncation=True, return_tensors="pt")
    return {
        "input_ids": inputs['input_ids'].tolist(),
        "attention_mask": inputs['attention_mask'].tolist(),
        "labels": inputs['labels'].tolist()
    }

# Apply tokenization function
tokenized_dataset = dataset.map(tokenize_translation, batched=True)

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x)
        if attention_mask is not None:
            # Apply attention mask to encoder outputs to ignore padding
            outputs = outputs * attention_mask.unsqueeze(-1)
        return outputs, hidden, cell

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_heads):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs, attention_mask=None):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # Apply attention on encoder outputs, using attention_mask
        if attention_mask is not None:
            attn_output, attn_weights = self.attention(output, encoder_outputs, encoder_outputs, key_padding_mask=~attention_mask.bool())
        else:
            attn_output, attn_weights = self.attention(output, encoder_outputs, encoder_outputs)

        # Concatenate attention output with LSTM output
        combined = torch.cat((attn_output, output), dim=2)
        prediction = self.fc(combined.squeeze(1))
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, attention_mask=None, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden, cell = self.encoder(src, attention_mask=attention_mask)
        outputs = []
        x = tgt[:, 0]  # Start token

        for t in range(1, tgt.size(1)):
            output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs, attention_mask=attention_mask)
            outputs.append(output.unsqueeze(1))
            x = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else output.argmax(1)

        return torch.cat(outputs, dim=1)

    def translate(self, src, attention_mask=None, max_length=50):
        self.eval()
        encoder_outputs, hidden, cell = self.encoder(src, attention_mask=attention_mask)
        x = torch.tensor([tokenizer.bos_token_id]).to(src.device)  # Start token for inference
        outputs = []

        for _ in range(max_length):
            output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs, attention_mask=attention_mask)
            x = output.argmax(1)
            outputs.append(x.item())
            if x.item() == tokenizer.eos_token_id:  # Stop if EOS token is generated
                break

        return outputs

# Instantiate model, criterion, optimizer
vocab_size = tokenizer.vocab_size
embed_size, hidden_size, num_heads = 128, 256, 4
encoder = LSTMEncoder(vocab_size, embed_size, hidden_size)
decoder = AttentionDecoder(vocab_size, embed_size, hidden_size, num_heads)
model = Seq2Seq(encoder, decoder)
model = torch.compile(model)

# Initialize weights
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param)
        else:
            nn.init.zeros_(param)

model.apply(init_weights)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define DataLoader with padding and attention_mask
def collate_fn(batch):
    src = torch.nn.utils.rnn.pad_sequence([torch.tensor(item['input_ids']) for item in batch],
                                          padding_value=tokenizer.pad_token_id, batch_first=True)
    tgt = torch.nn.utils.rnn.pad_sequence([torch.tensor(item['labels']) for item in batch],
                                          padding_value=tokenizer.pad_token_id, batch_first=True)
    attention_mask = (src != tokenizer.pad_token_id).type(torch.float32)
    return src, tgt, attention_mask

train_loader = DataLoader(
    tokenized_dataset['train'], batch_size=16, collate_fn=collate_fn,
    shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
scaler = GradScaler(device="cuda")  # Updated for the latest PyTorch syntax

accumulation_steps = 4  # Number of steps to accumulate gradients
smoothing = SmoothingFunction().method1  # Use smoothing method1 for BLEU score

for epoch in range(10):
    torch.cuda.empty_cache()
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")

    for i, (src, tgt, attention_mask) in enumerate(progress_bar):
        src, tgt, attention_mask = src.to(device), tgt.to(device), attention_mask.to(device)

        optimizer.zero_grad(set_to_none=True)  # Reset gradients

        with autocast(device_type="cuda"):  # Enable mixed-precision training
            output = model(src, tgt, attention_mask=attention_mask)
            output = output.view(-1, vocab_size)
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt) / accumulation_steps  # Scale loss

        scaler.scale(loss).backward()  # Backpropagate

        # Update weights every `accumulation_steps` batches
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix(loss=loss.item() * accumulation_steps)

    print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(train_loader)}")

    # Save model after each epoch
    torch.save(model.state_dict(), f"seq2seq_epoch_{epoch+1}.pth")
    print(f"Model saved for epoch {epoch + 1}")

    # Compute BLEU score for a small validation set with smoothing
    model.eval()
    bleu_scores = []
    for sample in tokenized_dataset['validation'][:10]:  # Evaluate on a small sample
        src = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
        attention_mask = (src != tokenizer.pad_token_id).type(torch.float32).to(device)
        reference = [tokenizer.decode(sample['labels'], skip_special_tokens=True).split()]
        predicted_ids = model.translate(src, attention_mask=attention_mask)
        hypothesis = tokenizer.decode(predicted_ids, skip_special_tokens=True).split()
        bleu_scores.append(sentence_bleu(reference, hypothesis, smoothing_function=smoothing))

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Epoch {epoch + 1}, Average BLEU Score with Smoothing: {avg_bleu}")
