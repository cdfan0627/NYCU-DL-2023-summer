
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataloader import read_bci_data
from models import EEGNet, DeepConvNet, ShallowcovNet

def main():
    activations = ["relu", "leakyrelu", "elu"]
    models = {"EEGNet": EEGNet, "DeepConvNet": DeepConvNet, "ShallowcovNet": ShallowcovNet}
    selected_model = "EEGNet"


    model_name = selected_model
    model_class = models[model_name]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data and process into Tensor format using dataloader.py
    _, _, test_data, test_label = read_bci_data()
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_label = torch.tensor(test_label, dtype=torch.long).to(device)

    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Test each best model on the test data
    for activation in activations:
        model = model_class(activation=activation).to(device)
        model_path = f"{model_name}_{activation}_best_model.pth"  # Replace with the actual path
        model.load_state_dict(torch.load(model_path))
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

            test_accuracy = correct_predictions / total_samples

        print(f"{model_name} activation function: {activation}, Test Accuracy: {100 * test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
