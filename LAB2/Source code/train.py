
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataloader import read_bci_data
from models import EEGNet, DeepConvNet, ShallowcovNet
import copy

def train_model(model, train_loader, test_loader, loss_function, optimizer, device, epochs):
    epoch_train_accuracy = []
    epoch_test_accuracy = []  # 用於儲存每個激活函數的測試準確度
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (correct_predictions / total_samples) *100
        epoch_train_accuracy.append(epoch_accuracy)

        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            test_accuracy = (correct_predictions / total_samples) * 100
        epoch_test_accuracy.append(test_accuracy)
        # Save the model with the highest test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    return epoch_train_accuracy, epoch_test_accuracy, best_model_wts
          

def main():
    activations = [ "relu", "leakyrelu", "elu"]
    models = {"EEGNet": EEGNet, "DeepConvNet": DeepConvNet, "ShallowcovNet": ShallowcovNet}
    selected_model = "DeepConvNet"

    model_name = selected_model
    model_class = models[model_name]
    

    epoch_train_accuracies = {}
    epoch_test_accuracies = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data and process into Tensor format using dataloader.py
    train_data, train_label, test_data, test_label = read_bci_data()
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_label = torch.tensor(train_label, dtype=torch.long).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_label = torch.tensor(test_label, dtype=torch.long).to(device)

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train models with different activation functions
    for activation in activations:
        print(f"Training {model_name} with activation function: {activation}")
        model = model_class(activation=activation).to(device)

        # Define batch size, learning rate, epochs, optimizer, and loss function
        learning_rate = 0.001
        epochs = 300
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()

        # Train the model and collect training accuracy trend
        epoch_train_accuracy, epoch_test_accuracy, best_model_wts = train_model(model, train_loader, test_loader, loss_function, optimizer, device, epochs)
        epoch_test_accuracies[activation] = epoch_test_accuracy
        epoch_train_accuracies[activation] = epoch_train_accuracy

        # Save the best model weights for this activation function
        torch.save(best_model_wts, f"{model_name}_{activation}_best_model.pth")  # Replace with the actual path

        print(f"{model_name} activation function: {activation}, Best Test Accuracy: {max(epoch_test_accuracy):.2f}")

    # Plot the accuracy trends for each activation function
    plt.figure(figsize=(10, 6))
    for activation, train_accuracies in epoch_train_accuracies.items():
        plt.plot(range(1, epochs+1), train_accuracies, label=f"{activation.capitalize()}_train")
    for activation, test_accuracies in epoch_test_accuracies.items():
        plt.plot(range(1, epochs+1), test_accuracies, label=f"{activation.capitalize()}_test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.title(f'Activation function comparision({model_name})')
    plt.show()

if __name__ == "__main__":
    main()
