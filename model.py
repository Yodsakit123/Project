import torchvision.models as models
import torch.nn as nn

def create_model(num_classes):
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model
