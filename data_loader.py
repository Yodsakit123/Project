import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split

def load_data(folder_name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Random flips
        transforms.RandomRotation(10),  # Random rotations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=folder_name, transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    imdsTrain, imdsValidation, imdsTest = random_split(dataset, [train_size, val_size, test_size])

    return dataset, imdsTrain, imdsValidation, imdsTest
