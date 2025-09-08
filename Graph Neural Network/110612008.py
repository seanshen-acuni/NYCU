import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
import ast
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x
# -----------------------------
# 資料讀取與處理
# -----------------------------
print("loading data...")
# 假設檔案路徑如下，請依實際需要調整
graph_path = "rebbit_graph.csv"
train_path = "train.csv"
public_test_path = "public_test.csv"
private_test_path = "private_test.csv"

# 讀取圖結構（無向圖）
graph_df = pd.read_csv(graph_path)
# 讀取 train 資料
train_df = pd.read_csv(train_path)
train_df['features'] = train_df['features'].apply(lambda x: ast.literal_eval(x))
train_features = np.array(train_df['features'].tolist(), dtype=float)
train_labels = train_df['label'].values
train_node_ids = train_df['node_id'].values

# 讀取 test 資料
public_test_df = pd.read_csv(public_test_path)
public_test_df['features'] = public_test_df['features'].apply(lambda x: ast.literal_eval(x))

private_test_df = pd.read_csv(private_test_path)
private_test_df['features'] = private_test_df['features'].apply(lambda x: ast.literal_eval(x))

test_df = pd.concat([public_test_df, private_test_df])
test_features = np.array(test_df['features'].tolist(), dtype=float)
test_node_ids = test_df['node_id'].values

# 收集所有唯一的 node_id
all_node_ids = set(graph_df['node1'].unique()).union(set(graph_df['node2'].unique()))
all_node_ids = set(train_node_ids).union(all_node_ids).union(set(test_node_ids))

# 建立排序後的 node_id 列表
unique_node_ids = sorted(list(all_node_ids))

# 建立映射字典
id_map = {old_id: new_id for new_id, old_id in enumerate(unique_node_ids)}

# 重新映射 edge_index
graph_df['node1'] = graph_df['node1'].map(id_map)
graph_df['node2'] = graph_df['node2'].map(id_map)

# 確保圖是無向的，添加反向邊
reverse_edges = graph_df[['node2', 'node1']]
graph_df = pd.concat([graph_df, reverse_edges], ignore_index=True)

edge_index = torch.tensor(graph_df[['node1', 'node2']].values.T, dtype=torch.long)

# 重新映射 train_node_ids 和 test_node_ids
train_node_ids_mapped = [id_map[node_id] for node_id in train_node_ids]
test_node_ids_mapped = [id_map[node_id] for node_id in test_node_ids]

train_node_ids = torch.tensor(train_node_ids_mapped, dtype=torch.long)
test_node_ids = torch.tensor(test_node_ids_mapped, dtype=torch.long)

# 更新 num_nodes 為唯一節點數量
num_nodes = len(unique_node_ids)
in_channels = 1433
hidden_channels = 128
out_channels = 47

# 將 train 與 test 特徵合併
x = np.zeros((num_nodes, in_channels))
x[train_node_ids] = train_features
x[test_node_ids] = test_features
x = torch.tensor(x, dtype=torch.float)

# labels
labels = torch.full((num_nodes,), -1, dtype=torch.long)
labels[train_node_ids] = torch.tensor(train_labels, dtype=torch.long)

# masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_node_ids] = True

#add feature normalization
x = (x - x[train_mask].mean(dim=0)) / x[train_mask].std(dim=0)

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_node_ids] = True

#public_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
#public_test_mask[test_node_ids[:500]] = True

#private_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
#private_test_mask[test_node_ids[500:]] = True

# 移動資料到 device 上
x = x.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
#public_test_mask = public_test_mask.to(device)
#private_test_mask = private_test_mask.to(device)
print("data loading complete")

# -----------------------------
# 建立與訓練模型
# -----------------------------
print("Starting model training...")
model = GNNModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
num_epochs = 95
train_accuracies = []
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    out = model(x, edge_index)
    loss = criterion(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    
    # 計算訓練準確度
    pred = out[train_mask].argmax(dim=1)
    correct = (pred == labels[train_mask]).sum().item()
    total_samples = train_mask.sum().item()
    train_accuracy = correct / total_samples


    losses.append(loss.item())
    train_accuracies.append(train_accuracy)

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}")
print("Model training complete")
torch.save(model, "110612008.pth")
# -----------------------------
# 預測與產生 Submission
# -----------------------------
model.eval()
with torch.no_grad():
    out = model(x, edge_index)
    pred = out.argmax(dim=1).cpu().numpy()

inverse_id_map = {v: k for k, v in id_map.items()}
original_test_node_ids = [inverse_id_map[idx] for idx in test_node_ids_mapped]
test_pred_labels = []

for idx in tqdm(original_test_node_ids, desc="Predicting test nodes"):
    test_pred_labels.append(pred[idx])

test_submission_df = pd.DataFrame({
    "node_id": original_test_node_ids,
    "label": test_pred_labels
})
test_submission_df.to_csv("110612008.csv", index=False)

print("result saved to 110612008.csv")