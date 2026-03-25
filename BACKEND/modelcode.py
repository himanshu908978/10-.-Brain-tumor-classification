# THIS CODE IS WRITTEN IN GOOGLE COLAB AND MODEL TRAINED IN THE GOOGLE COLAB NOTEBOOK AND DATASET IN IN GOOGLE DRIVE

'''
import random
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import Counter
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torchvision.models import DenseNet121_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"




def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)




def tumor_dataset(data_path, input = 224):
    train_tf = transforms.Compose([
    transforms.Resize((input,input)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
    ])

    test_tf = transforms.Compose([
    transforms.Resize((input,input)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
    ])

    data_dir = Path(data_path)
    full_dataset = datasets.ImageFolder(str(data_dir))

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_subset, test_subset = random_split(
      full_dataset,
      [train_size, test_size],
      generator=torch.Generator().manual_seed(42)
    )

    train_indices = train_subset.indices
    test_indices = test_subset.indices

    # train_dataset, test_dataset = random_split(full_dataset, [training_size, testing_size], generator = torch.Generator().manual_seed(42))
    train_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(str(data_dir), transform=train_tf),
    train_indices
    )

    test_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(str(data_dir), transform=test_tf),
    test_indices
    )

    train_loader = DataLoader(train_dataset,batch_size = 8, shuffle = True, num_workers = 2, pin_memory = True)
    test_loader = DataLoader(test_dataset,batch_size = 8, shuffle = False, num_workers = 2, pin_memory = True)

    return train_loader, test_loader, full_dataset




def build_model(classes = 2, feature_extract = False):
  model1 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

  if feature_extract :
    for param in model1.parameters():
      param.requires_grad = False

  input_size = model1.classifier.in_features
  model1.classifier = nn.Linear(input_size, classes)
  return model1




def train_model(model, train_loader, save_path, class_weights, device, lr = 1e-4, epoch = 10):
  save_path = Path(save_path)
  save_path.mkdir(parents = True, exist_ok = True)
  model = model.to(device)
  param_to_update = [p for p in model.parameters() if p.requires_grad]
  optimizer = optim.Adam(param_to_update,lr = lr)
  
  criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))
  best_f1 = 0.0
  for i in range(epoch):
    model.train()
    print("\n")
    print("\n")
    print("\n")
    print("-"*20)
    print(f"Epoch :- {i+1}/{epoch}")
    print("-"*20)
    print("\n")
    since = time.time()
    total_time = 0
    total_loss = 0
    y_true = []
    y_pred = []
    for batch, label in train_loader:
      batch = batch.to(device)
      label = label.to(device)
      output = model(batch)
      loss = criterion(output, label)

      _,pred = torch.max(output,1)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      y_true.extend(label.detach().cpu().numpy().tolist())
      y_pred.extend(pred.detach().cpu().numpy().tolist())

    f1 = f1_score(y_true, y_pred,average = 'macro',zero_division = 0)
    print(f"\nLoss :- {total_loss / len(train_loader)} \n Accuracy :- {accuracy_score(y_true,y_pred)} \n F1 :- {f1} \n Recall :- {recall_score(y_true, y_pred, average = 'macro', zero_division = 0)}")
    if best_f1 < f1:
      best_f1 = f1
      torch.save(model.state_dict(), save_path/'best_model_epoch.pth')
      print(f"Model saved at :- '{save_path}/best_model_epoch.pth'")
    total_time = time.time() - since
    print(f"\n Estimated Time in epoch {i} training :- {total_time}\n")

  return model



def test_model(model, test_loader, class_weights, device):

  model = model.to(device)
  y_true = []
  y_pred = []
  loss_list = []
  criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))

  with torch.no_grad():
    model.eval()
    for batch,label in test_loader:
      batch = batch.to(device)
      label = label.to(device)

      output = model(batch)
      loss = criterion(output, label)

      value, indicies = torch.max(output, 1)

      y_true.extend(label.detach().cpu().numpy().tolist())
      y_pred.extend(indicies.detach().cpu().numpy().tolist())
      loss_list.append(loss.item())

    print(f"\n Some Important metrics in testing phase :- \n 1).Loss :- {sum(loss_list) / len(loss_list)} \n 2).Accuracy :- {accuracy_score(y_true,y_pred)} \n 3).F1 :- {f1_score(y_true,y_pred,average = 'macro', zero_division = 0)} \n 4).Recall :- {recall_score(y_true, y_pred, average = 'macro', zero_division = 0)}")




def main(data_path, save_path, device, seed = 42, classes = 2):
  set_seed(seed)
  train_loader, test_loader, full_dataset = tumor_dataset(data_path)
  model = build_model(classes = classes)

  class_names = full_dataset.classes
  print("Class names in dataset = ", class_names)

  labels = [l for _, l in full_dataset] # 5002
  labels_count = Counter(labels) # [2501, 2501]
  total = sum(labels_count.values()) # 5002
  num_classes = len(labels_count) # 2
  weights = []

  for i in range(num_classes):
    weight = total / (num_classes * labels_count[i])
    weights.append(weight)

  weights = torch.tensor(weights,dtype = torch.float).to(device)
  model = train_model(model, train_loader, save_path, class_weights = weights, device = device)
  test_model(model, test_loader, class_weights = weights, device = device)
  torch.save(model.state_dict(),Path(save_path)/'final_model.pth')
  return model



'''