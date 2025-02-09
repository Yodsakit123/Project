import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model, train_loader, num_epochs=50):
    # Automatically detect device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            # Move inputs & labels to the same device as model
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] completed")

    return model

