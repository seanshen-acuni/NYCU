#python introToAI_HW3_cuda_edition02.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labeled_data_path = "datasets/labeled_data"
unlabeled_data_path = "datasets/unlabeled_data"
private_test_data_path = "datasets/private_test_data"
class_to_label = {
    "dog"  :0,
    "wolf" :1,
    "cat"  :2,
    "fox"  :3,
    "sheep":4,
    "goat" :5,
    "horse":6,
    "deer" :7,
    "bird" :8,
    "duck" :9
}
label_l = []
image_list_l = []
image_list_unl = []
image_list_pri = []

image_files_unl = [f for f in os.listdir(unlabeled_data_path) if f.endswith('.jpg')]
image_files_pri = [f for f in os.listdir(private_test_data_path) if f.endswith('.jpg')]

for class_name, label in class_to_label.items():
    class_folder = os.path.join(labeled_data_path, class_name)
    if os.path.exists(class_folder):
        for image_file in os.listdir(class_folder):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(class_folder, image_file)
                image_list_l.append(image_path)
                label_l.append(label)

labeled_data = pd.DataFrame({"image_path": image_list_l,"label": label_l})

for image_file in image_files_unl:
    image_path_unl = os.path.join(unlabeled_data_path, image_file)
    image_list_unl.append(image_path_unl)

for image_file in image_files_pri:
    image_path_pri = os.path.join(private_test_data_path, image_file)
    image_list_pri.append(image_path_pri)

unlabeled_data = pd.DataFrame(image_list_unl, columns=["image_path"])
private_test_data = pd.DataFrame(image_list_pri, columns=["image_path"])

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, labeled=True):
        self.dataframe = dataframe
        self.transform = transform
        self.labeled = labeled
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.labeled:
            label = self.dataframe.iloc[idx, 1]
            return image, label
        return image

# Load labeled and unlabeled data
labeled_dataset = ImageDataset(labeled_data, transform=transform, labeled = True)
unlabeled_dataset = ImageDataset(unlabeled_data, transform=transform, labeled=False)
private_test_dataset = ImageDataset(private_test_data, transform=transform, labeled=False)

labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)
private_test_loader = DataLoader(private_test_dataset, batch_size=32, shuffle=False)

# Define an RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(-1))
        _, (hn, _) = self.rnn(x)
        out = self.fc(hn[-1])
        return out

input_size = 128
hidden_size = 128
output_size = 10 # 10 categories
model = RNNModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def generate_pseudo_labels(model, dataloader):
    pseudo_labels = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Generating pseudo-labels"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            pseudo_labels.extend(predicted.cpu().numpy())
    return pseudo_labels
# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0 
    for images, labels in tqdm(labeled_loader, desc = f"Epoch {epoch + 1} - Labeled"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Labeled Loss: {total_loss:.4f}")

    print("Generating pseudo-labels for unlabeled data")
    pseudo_labels = generate_pseudo_labels(model, unlabeled_loader)
    pseudo_labeled_data = pd.DataFrame({
        "image_path": unlabeled_data["image_path"],  # Use the original paths from unlabeled_data
        "label": pseudo_labels  # Assign the pseudo-labels
    })
    train_data = pd.concat([labeled_data, pseudo_labeled_data], ignore_index=True)
    train_dataset = ImageDataset(train_data, transform=transform, labeled=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Combined"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Combined Loss: {total_loss:.4f}")
torch.save(model, "model.pth")
# Prediction on the entire dataset
labeled_data_clean = labeled_data[['image_path']]
test_data_k = pd.concat([labeled_data_clean, unlabeled_data, private_test_data], ignore_index=False)
test_dataset_k = ImageDataset(test_data_k, transform=transform, labeled=False)
test_loader_k = DataLoader(test_dataset_k, batch_size=32, shuffle=False)
test_data = pd.concat([unlabeled_data, private_test_data], ignore_index=False)
test_dataset = ImageDataset(test_data, transform=transform, labeled=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model.eval()
predictions_k = []
with torch.no_grad():
    for images in tqdm(test_loader_k, desc = "Predicting for kaggle submission"):
        images = images.to(device)
        outputs = model(images)
        _, predicted_k = torch.max(outputs, 1)
        predictions_k.extend(predicted_k.cpu().numpy())
test_data_k["image_path"] = test_data_k["image_path"].apply(os.path.basename)
# Save to sample_submission.csv
test_data_k['predictions'] = predictions_k
test_data_k.rename(columns={"image_path": "ID", "predictions": "label"}, inplace=True)
test_data_k['sort_key'] = test_data_k['ID'].str.extract(r'(\d+)').astype(int)
test_data_k = test_data_k.sort_values('sort_key')
test_data_k = test_data_k.drop('sort_key', axis=1)
test_data_k[["ID", "label"]].to_csv('sample_submission_kaggle.csv', index=False)
predictions = []
with torch.no_grad():
    for images in tqdm(test_loader, desc = "Predicting for e3 submission"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
test_data["image_path"] = test_data["image_path"].apply(os.path.basename)
# Save to sample_submission.csv
test_data['predictions'] = predictions
test_data.rename(columns={"image_path": "ID", "predictions": "label"}, inplace=True)
test_data['sort_key'] = test_data['ID'].str.extract(r'(\d+)').astype(int)
test_data = test_data.sort_values('sort_key')
test_data = test_data.drop('sort_key', axis=1)
test_data[["ID", "label"]].to_csv('sample_submission.csv', index=False)
print("result saved")

