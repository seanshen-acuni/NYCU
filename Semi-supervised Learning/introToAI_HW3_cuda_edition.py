#python introToAI_HW3_cuda_edition.py
import os
import gc
import torch
import random
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import glob
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labeled_data_path = "datasets/labeled_data"
unlabeled_data_path = "datasets/unlabeled_data"
private_test_data_path = "datasets/private_test_data"
batch_size = 128
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = glob.glob(f"{self.root_dir}/*.*")  # 匹配所有圖像文件
        self.image_names = [os.path.basename(path) for path in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image.to(device), torch.tensor(-1, dtype = torch.long, device = device)

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset, private_test_dataset, transform=None):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.private_test_dataset = private_test_dataset
        self.transform = transform
        self.all_samples = self.get_all_samples()
    def get_all_samples(self):
        labeled_samples = [(data[0], torch.tensor(data[1], dtype=torch.long, device=device)) for data in self.labeled_dataset]
        unlabeled_samples = [(path, torch.tensor(-1, dtype = torch.long, device=device)) for path in self.unlabeled_dataset.image_paths]
        private_test_samples = [(path, torch.tensor(-1, dtype = torch.long, device=device)) for path in self.private_test_dataset.image_paths]

        return labeled_samples + unlabeled_samples + private_test_samples
    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample_path, target = self.all_samples[idx]
        if isinstance(sample_path, torch.Tensor):
            image = sample_path
        else:
            image = Image.open(sample_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image.to(device), target

class SemiSupervisedDataset(Dataset):
    # features  = the features of the data where probability > 0.9
    # the feature size = batch x 3 x 32 x 32
    # indexes = indexes where the probability of 0.9 or higher happened
    def __init__(self, features, labels):
        self.data = features.to(device)
        self.labels = labels.clone().detach().to(device)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_epoch(epoch, training_loader, loss_func, optimizer, model, batch_size=batch_size):
    
    # due to time, only train the first 10 of the epoch = 64 * 1000 =64000 samples per epoch 
    model.train()
    running_loss = 0
    for i, (inputs, labels) in enumerate(training_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 0:
            print(f'index {i}, loss {running_loss}')

    return running_loss/batch_size

def valid_epoch(test_loader, model):
    
    dataset_list = []
    model.eval()

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # softmax the outputs. For probability output
            softmax = nn.Softmax(dim = -1)
            softmax_outputs = softmax(outputs)

            # get the values of the outputs. val = maximum softmax output, idx = idx of the maximum value 
            val, idx = torch.max(softmax_outputs, dim = -1)

            # semi_data_boolean = T/F boolean tensor where is True if probability is > 0.9
            semi_data_boolean = val > 0.95
            high_confidence_inputs = inputs[semi_data_boolean]
            high_confidence_preds = idx[semi_data_boolean]
    
            if len(high_confidence_preds) > 0:
                semi_dataset = SemiSupervisedDataset(high_confidence_inputs, high_confidence_preds)
                dataset_list.append(semi_dataset)

    return dataset_list

def predict_and_save(model, dataloader, predictions, dataset):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for idx, pred in enumerate(preds):
                sample_path_or_image = dataset.all_samples[i * batch_size + idx][0]
                if isinstance(sample_path_or_image, torch.Tensor):
                    sample_id = f"sample_{i * batch_size + idx}"
                else:
                    sample_id = os.path.basename(sample_path_or_image)
                predictions.append([sample_id, pred.item()])

def main():
    
    def collate_fn(batch):
        images = torch.stack([item[0].to(device) for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long).to(device)
        return images, labels

    batch_size = 128
    labeled_data = datasets.ImageFolder(root = labeled_data_path, transform = transform)
    labeled_dataloader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    unlabeled_data = CustomDataset(root_dir = unlabeled_data_path, transform = transform)
    private_test_data = CustomDataset(root_dir = private_test_data_path, transform = transform)
    
    model = models.resnet18()
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
    model = model.to(device)

    # train this model using Adam and the train set. output the test data accuracy
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # carry out the training.
    num_epoch = 1
    for epoch in range(num_epoch):
        print(f'epoch is {epoch + 1} / {num_epoch}')
         # trian the model for the first epoch

        train_epoch(
            epoch = epoch,
            training_loader = labeled_dataloader, 
            loss_func = loss_fn,
            model = model, 
            optimizer = optimizer,
            batch_size=batch_size
            )
        # then validate the model for the first epoch
        unlabeled_dataloader = DataLoader(unlabeled_data, batch_size = batch_size, shuffle = True)
        dataset_list = valid_epoch(test_loader = unlabeled_dataloader,model = model)
        if dataset_list: 
            labeled_data = ConcatDataset([labeled_data] + dataset_list)
            labeled_dataloader = DataLoader(labeled_data, batch_size = batch_size, shuffle = True,
                                        collate_fn = collate_fn, num_workers = 0)
    combined_dataset = CombinedDataset(labeled_data, unlabeled_data, private_test_data, transform = transform)
    combined_dataloader = DataLoader(combined_dataset, batch_size = batch_size, shuffle = False)
    predictions = []
    predict_and_save(model, combined_dataloader, predictions, combined_dataset)
    csv_filename = "sample_submission.csv"
    with open(csv_filename, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "label"])
        writer.writerows(predictions)
    print(f"result saved to {csv_filename}")
    
if __name__ == "__main__":
    main()