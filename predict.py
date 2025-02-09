import torch
import numpy as np

def predict_model(model, test_loader, class_names):
    model.eval()
    YTest_scores = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            YTest_scores.extend(outputs.numpy())

    YTest = np.argmax(YTest_scores, axis=1)
    return [class_names[i] for i in YTest]
