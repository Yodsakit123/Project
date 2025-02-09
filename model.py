import torchvision.models as models
import torch.nn as nn
from torchvision.models import SqueezeNet1_0_Weights

def create_model(num_classes):
    # Load SqueezeNet with updated weights syntax
    model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)

    # Freeze earlier layers (transfer learning)
    for param in model.features.parameters():
        param.requires_grad = False  

    # Replace classifier layer to match number of classes
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.classifier.add_module("activation", nn.Softmax(dim=1))  # Softmax for multi-class

    return model
