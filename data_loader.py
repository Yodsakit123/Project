import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split

def load_data(folder_name):
    # Define transforms: Resize, Convert to Tensor, Normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to match input size
        transforms.ToTensor(),          # Convert images to PyTorch Tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load dataset with transform
    dataset = datasets.ImageFolder(root=folder_name, transform=transform)

    # Split dataset (Train 70%, Validation 15%, Test 15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    imdsTrain, imdsValidation, imdsTest = random_split(dataset, [train_size, val_size, test_size])

    return dataset, imdsTrain, imdsValidation, imdsTest
