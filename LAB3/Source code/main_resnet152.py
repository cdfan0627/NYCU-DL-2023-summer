import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from ResNet import ResNet18, ResNet50, ResNet152
from dataloader_resnet152 import LeukemiaLoader
import time

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

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_accuracy = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        total_accuracy += accuracy_score(target.cpu().numpy(), pred.cpu().numpy())
    return total_accuracy / len(train_loader)


def evaluate(model, valid_loader, device):
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            total_accuracy += accuracy_score(target.cpu().numpy(), pred.cpu().numpy())
    return total_accuracy / len(valid_loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [ResNet152()]
    model_names = ['resnet_152']
    train_loader = DataLoader(LeukemiaLoader('train',None), batch_size=32, shuffle=True, num_workers=1)
    valid_loader = DataLoader(LeukemiaLoader('valid',None), batch_size=32, shuffle=False, num_workers=1)
    

    for model, model_name in zip(models, model_names):
        print()
        print(model_name)
        model = model.to(device)
        best_valid_acc = 0
        optimizer = optim.SGD(model.parameters(), lr=0.0005, weight_decay=0.0001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        epochs = 550
        train_accs, valid_accs = [], []

        for epoch in range(epochs):
            start_time = time.time()
            train_acc = train(model, train_loader, criterion, optimizer, device)
            valid_acc = evaluate(model, valid_loader, device)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc 
                torch.save(model.state_dict(), f'best_model_{model_name}.pth')  
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f'Epoch {epoch+1}/{epochs}: train_acc = {train_acc}, valid_acc = {valid_acc} epoch time: {epoch_time / 60} ')

        plt.plot(range(1, epochs+1), train_accs, label=f'{model_name} Train')
        plt.plot(range(1, epochs+1), valid_accs, label=f'{model_name} Valid')


    plt.title(f'Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()