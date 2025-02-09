from data_loader import load_data
from model import create_model
from train import train_model
from predict import predict_model
from utils import plot_confusion_matrix
from torch.utils.data import DataLoader

# Load dataset
dataset, imdsTrain, imdsValidation, imdsTest = load_data("testimage")

# Create DataLoaders
train_loader = DataLoader(imdsTrain, batch_size=32, shuffle=True)
test_loader = DataLoader(imdsTest, batch_size=32, shuffle=False)

# Create & Train Model
model = create_model(len(dataset.classes))
model = train_model(model, train_loader)

# Predict Labels
YTest = predict_model(model, test_loader, dataset.classes)

# Plot Confusion Matrix
TTest = [label for _, label in imdsTest]  # Extract true labels
plot_confusion_matrix(TTest, YTest, dataset.classes)
