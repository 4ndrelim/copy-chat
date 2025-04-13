import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import re
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

# === Tokenizer ===
class WordTokenizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SEP>": 2,
            "<SOS>": 3,
            "<EOS>": 4,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self, texts):
        word_freq = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_freq.update(tokens)
        for word, freq in word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        if add_special_tokens:
            ids = [self.word2idx["<SOS>"]] + ids + [self.word2idx["<EOS>"]]
        return ids

    def decode(self, ids):
        return " ".join(self.idx2word.get(i, "<UNK>") for i in ids if i != self.word2idx["<PAD>"])

# === LSTM Language Model ===
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

# === Dataset Preparation ===
def prepare_dataset(df, tokenizer, max_input_len=30, max_output_len=30):
    data = []
    for _, row in df.iterrows():
        inp = f"{row['prefix']} <SEP> {row['sentiment']}"
        out = row['completion']
        x = tokenizer.encode(inp)
        y = tokenizer.encode(out)
        x = x[:max_input_len] + [0] * (max_input_len - len(x))
        y = y[:max_output_len] + [0] * (max_output_len - len(y))
        data.append((x, y))
    return torch.tensor([x for x, _ in data]), torch.tensor([y for _, y in data])

# === Generation ===
def generate_completion(model, tokenizer, prompt, max_len=30):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    generated = []

    hidden = None
    for _ in range(max_len):
        logits, hidden = model(input_tensor, hidden)
        next_token = torch.argmax(logits[0, -1]).item()
        if next_token == tokenizer.word2idx["<EOS>"]:
            break
        generated.append(next_token)
        input_tensor = torch.tensor([[next_token]], dtype=torch.long)
    return tokenizer.decode(generated)

# === Training ===
def train_lstm_model(train_file, test_file, output_file):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Auto-generate prefix and completion from full text
    def create_prefix_completion(text, prefix_len=6):
        words = text.split()
        prefix = " ".join(words[:prefix_len])
        completion = " ".join(words[prefix_len:])
        return prefix, completion

    df_train = df_train[df_train['text'].notna()]
    df_train[['prefix', 'completion']] = df_train['text'].apply(lambda x: pd.Series(create_prefix_completion(x)))
    df_train['sentiment'] = df_train['sentiment'].fillna("Neutral")

    tokenizer = WordTokenizer()
    tokenizer.build_vocab(df_train['completion'].tolist() + df_train['prefix'].tolist())

    x_train, y_train = prepare_dataset(df_train, tokenizer)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMLanguageModel(len(tokenizer.word2idx))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Training LSTM...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(loader):
            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    print("Generating on test set...")
    results = []
    for _, row in df_test.iterrows():
        input_str = f"{row['prefix']} <SEP> {row['sentiment']}"
        completion = generate_completion(model, tokenizer, input_str)
        results.append({
            "id": row["id"],
            "prefix": row["prefix"],
            "sentiment": row["sentiment"],
            "completion": completion
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

# === Run ===
if __name__ == "__main__":
    train_path = "dataset/mimicry/annotated/trump_train.csv"
    test_path = "dataset/mimicry/annotated/trump_test.csv"
    output_path = "predictions/mimicry/trump_lstm_baseline.csv"
    train_lstm_model(train_path, test_path, output_path)
