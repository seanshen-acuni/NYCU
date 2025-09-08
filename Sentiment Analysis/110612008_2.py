#python 110612008_2.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tokenizer import Tokenizer  # 使用您已經定義的 Tokenizer
from torch.utils.data import DataLoader, TensorDataset
import json
from tqdm import tqdm
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print("Loading Data...")
# 初始化 Tokenizer
tokenizer = Tokenizer(vocab='token_to_index.pkl', max_length=53)

# 讀取訓練資料
label_to_int = {'neutral': 0, 'positive': 1, 'negative': 2}
int_to_label = {0: 'neutral', 1: 'positive', 2: 'negative'}

train_texts = []
train_labels = []

with open('dataset/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        train_data = json.loads(line)
        train_texts.append(train_data['text'])
        train_labels.append(label_to_int[train_data['label']])

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
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
dataloader_unshuffle = DataLoader(dataset, batch_size=512, shuffle=False)
print("Training...")
# 定義模型參數
vocab_size = tokenizer.get_vocab_size()
embedding_dim = 512
hidden_dim = 1024
output_dim = 3  # 假設標籤是從 0 開始的整數

# 初始化模型、損失函數和優化器
model = RNN_Model(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 混淆矩陣函數
def confusion_matrix(pred, original):  # pass predicted and original labels to this function
    matrix = np.zeros((3, 3))  # 3x3 matrix for neutral, positive, negative
    for i in range(len(pred)):  
        matrix[int(original[i]), int(pred[i])] += 1  # Count occurrences
    
    # 計算評估指標
    precision = np.diag(matrix) / (np.sum(matrix, axis=0) + 1e-10)
    recall = np.diag(matrix) / (np.sum(matrix, axis=1) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return matrix

# 繪製混淆矩陣
def plot_confusion_matrix(matrix, class_labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix.astype(int), annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion_matrix.png")
    plt.show(block=False)
    print("Confusion Matrix Plot saved")
# 訓練模型
num_epochs = 50
epoch_losses = []
epoch_accuracies = []
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0
    for batch_inputs, batch_labels in dataloader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)
    accuracy = 100 * correct_predictions / total_samples
    epoch_losses.append(total_loss)
    epoch_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
    if 1.7 < total_loss < 1.9:
        print(f"Early stopping at {epoch + 1}")
        break

epochs = range(1, len(epoch_losses) + 1)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
#Loss plot
ax[0].plot(epochs, epoch_losses, label='Loss', color='red')
ax[0].set_title('Epoch vs Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
#Accuracy plot
ax[1].plot(epochs, epoch_accuracies, label='Accuracy', color='red')
ax[1].set_title('Epoch vs Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
plt.tight_layout()
plt.savefig("Loss_Accuracy_history_plot.png")
plt.show(block=False)
print("Loss/Accuracy History Plot saved")
torch.save(model, "model.pth")
print("Model saved")

# 預測函數
def predict(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        sequence = tokenizer.encode(text)
        input_tensor = torch.tensor([sequence], dtype=torch.long)
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class
print("Loading test data...")
# 測試模型
test_ids = []
test_texts = []
with open('dataset/test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        test_data = json.loads(line)
        test_texts.append(test_data['text'])
        test_ids.append(test_data['id'])
test_sequences = tokenizer.batch_encode(test_texts)
test_padded = np.array(test_sequences)
test_inputs = torch.tensor(test_padded, dtype=torch.long)
test_dataset = TensorDataset(test_inputs)
test_dataloader = DataLoader(test_dataset, batch_size=512)
#test_text = "It seems that using Twitter has become quite an adventure for many users these days. Well, that's interesting, I suppose."
print("Predicting...")
predictions_train = []
true_train = []
with torch.no_grad():
    for batch_inputs, batch_labels in tqdm(dataloader_unshuffle, desc="Predicting on train data"):
        batch_inputs = batch_inputs.to(device)
        outputs = model(batch_inputs)
        _, predicted = torch.max(outputs, 1)
        predictions_train.extend(predicted.cpu().numpy())
        true_train.extend(batch_labels.cpu().numpy())
conf_matrix = confusion_matrix(np.array(predictions_train), np.array(true_train))
plot_confusion_matrix(conf_matrix, class_labels=["Neutral", "Positive", "Negative"])
predictions = []
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Predicting on test data"):
        batch_inputs = batch[0].to(device)
        outputs = model(batch_inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

predicted_labels = [int_to_label[label] for label in predictions]

test_df = pd.DataFrame({"id": test_ids, "label": predicted_labels})
test_df[['id', 'label']].to_csv('sample_submission.csv', index=False)
print("Result saved to sample_submission.csv")
