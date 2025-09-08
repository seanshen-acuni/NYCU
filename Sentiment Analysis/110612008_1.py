#python 110612008.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tokenizer import Tokenizer  # 使用您已經定義的 Tokenizer
from torch.utils.data import DataLoader, TensorDataset

# 定義 RNN 模型（不需要 CNN 編碼器，因為我們處理的是文本）
class RNN_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        output, hidden = self.rnn(embedded)  # output: (batch_size, seq_length, hidden_dim)
        hidden = self.dropout(hidden[-1])  # (batch_size, hidden_dim)
        out = self.fc(hidden)  # (batch_size, output_dim)
        return out

# 初始化 Tokenizer
tokenizer = Tokenizer(vocab='token_to_index.pkl', max_length=100)

# 讀取訓練資料
import json
label_to_int = {'neutral': 0, 'positive': 1, 'negative': 2}
int_to_label = {0: 'neutral', 1: 'positive', 2: 'negative'}

train_texts = []
train_labels = []

with open('dataset/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        train_texts.append(data['text'])
        train_labels.append(label_to_int[data['label']])

# 將文本轉換為索引序列
train_sequences = tokenizer.batch_encode(train_texts)
train_padded = np.array(train_sequences)

# 將標籤轉換為數組
train_labels = np.array(train_labels)

# 將數據轉換為 PyTorch 張量
train_inputs = torch.tensor(train_padded, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.long)

# 建立資料集和資料加載器
dataset = TensorDataset(train_inputs, train_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定義模型參數
vocab_size = tokenizer.get_vocab_size()
embedding_dim = 128
hidden_dim = 256
output_dim = 3  # 假設標籤是從 0 開始的整數

# 初始化模型、損失函數和優化器
model = RNN_Model(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 訓練模型
num_epochs = 5  # 訓練輪數

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_inputs, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 預測函數
def predict(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        sequence = tokenizer.encode(text)
        input_tensor = torch.tensor([sequence], dtype=torch.long)
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

# 測試模型
test_text = "It seems that using Twitter has become quite an adventure for many users these days. Well, that's interesting, I suppose."
predicted_label = int_to_label[predict(test_text, model, tokenizer)]
print(f"輸入文本: {test_text}")
print(f"預測標籤: {predicted_label}")
