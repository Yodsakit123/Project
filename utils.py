import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(TTest, YTest, class_names):
    conf_matrix = confusion_matrix(TTest, YTest)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels"), plt.ylabel("True Labels"), plt.title("Confusion Matrix")
    plt.show()
