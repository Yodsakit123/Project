from data_loader import load_data
from model import create_model
from train import train_model
from predict import predict_model
from utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# Load dataset
dataset, imdsTrain, imdsValidation, imdsTest = load_data("testimage")

# Debug: Print classes
print("Classes found:", dataset.classes)
print("Number of classes:", len(dataset.classes))

if len(dataset.classes) < 2:
    raise ValueError("Error: At least 2 classes are required for classification!")

# Create DataLoaders
train_loader = DataLoader(imdsTrain, batch_size=32, shuffle=True)
test_loader = DataLoader(imdsTest, batch_size=32, shuffle=False)

# Create & Train Model
model = create_model(len(dataset.classes))
model = train_model(model, train_loader)

# Predict Labels
YTest = predict_model(model, test_loader, dataset.classes)

# Convert True Labels (TTest) to Strings
TTest = [dataset.classes[label] for _, label in imdsTest]

# Convert Predicted Labels (YTest) to Strings (If Needed)
if isinstance(YTest[0], int):
    YTest = [dataset.classes[label] for label in YTest]

# Debugging: Print Samples
print("TTest Sample:", TTest[:5])
print("YTest Sample:", YTest[:5])

# Calculate Accuracy
accuracy = accuracy_score(TTest, YTest)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Plot Confusion Matrix
plot_confusion_matrix(TTest, YTest, dataset.classes)
