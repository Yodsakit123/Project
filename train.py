import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model, train_loader, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")  # Move to GPU if available
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

