import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(TTest, YTest, class_names, save_path="confusion_matrix.png"):
    conf_matrix = confusion_matrix(TTest, YTest)

    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)  # Save the plot as an image
    print(f"Confusion matrix saved as {save_path}")

