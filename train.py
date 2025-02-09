import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate for stability
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropy for multi-class classification

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Standard multi-class loss
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")

    return model
