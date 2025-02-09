import torch
import numpy as np

def predict_model(model, test_loader, class_names):
    # Automatically detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model.to(device)
    model.eval()

    YTest_scores = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)  
            outputs = model(inputs)
            YTest_scores.extend(outputs.cpu().numpy())  

    YTest = np.argmax(YTest_scores, axis=1)
    return [class_names[i] for i in YTest]
