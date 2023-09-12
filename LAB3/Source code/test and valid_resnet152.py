import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from dataloader_resnet152 import LeukemiaLoader
from ResNet import ResNet18, ResNet50, ResNet152
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


# Load the model
def load_model(model_path, model_type):
    if model_type == 'resnet_18':
        model = ResNet18()
    elif model_type == 'resnet_50':
        model = ResNet50()
    elif model_type == 'resnet_152':
        model = ResNet152()
    else:
        print("Invalid model type")
        return

    model.load_state_dict(torch.load(model_path))
    return model

def predict(model, test_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)
            all_preds.extend(preds.cpu().numpy().tolist())
    all_preds = [item for sublist in all_preds for item in sublist]
    return all_preds

# Evaluate the model on valid data
def evaluate(model, valid_loader, device):
    model.eval()
    total_accuracy = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            total_accuracy += accuracy_score(target.cpu().numpy(), pred.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return total_accuracy / len(valid_loader), all_targets, all_preds

def save_result(model_name, csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv(f"./312551093_{model_name}.csv", index=False)

# Function to compute and plot normalized confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=[0, 1])
    disp = disp.plot(include_values=True,
                     cmap='viridis', ax=None, xticks_rotation='horizontal')
    plt.title(f'Confusion Matrix of {model_name}_Original')
    plt.show()

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ['resnet_152']
    valid_loader = DataLoader(LeukemiaLoader('valid',None), batch_size=32, shuffle=False)

    for model_name in model_names:
        model_path = f'best_model_{model_name}.pth'
        model = load_model(model_path, model_name)
        model = model.to(device)

        # Evaluate on validation data
        valid_acc, all_targets, all_preds = evaluate(model, valid_loader, device)
        print(f'{model_name}: valid_acc = {valid_acc}')

        # Plot confusion matrix
        plot_confusion_matrix(all_targets, all_preds, model_name)

        # Predict on test data
        test_loader = DataLoader(LeukemiaLoader('test',model_name), batch_size=32, shuffle=False)
        csv_path = (f'{model_name}_test.csv')
        test_preds = predict(model, test_loader, device)
        save_result(model_name, csv_path, test_preds)

if __name__ == "__main__":
    main()