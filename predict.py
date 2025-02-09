import torch
import numpy as np

def predict_model(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    YTest_scores = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            YTest_scores.extend(outputs.cpu().numpy())  # Move to CPU for processing

    # Debugging: Print raw outputs
    print("Raw YTest_scores:", YTest_scores[:5])

    if len(YTest_scores) == 0:
        print("Warning: No predictions were made!")
        return []

    # Convert softmax scores to class indices
    YTest_indices = np.argmax(YTest_scores, axis=1)

    # Convert indices to class names
    YTest_labels = [class_names[i] for i in YTest_indices]

    # Debugging: Print final predicted labels
    print("Final Predicted Labels:", YTest_labels[:5])

    return YTest_labels
