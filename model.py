import torchvision.models as models
import torch.nn as nn

def create_model(num_classes):
    model = models.squeezenet1_0(pretrained=True)

    # Freeze earlier layers (Recommended for transfer learning)
    for param in model.features.parameters():
        param.requires_grad = False  

    # Replace classifier layer to match number of classes
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.classifier.add_module("activation", nn.Softmax(dim=1))  # Softmax for multi-class

    return model
