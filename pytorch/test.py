import os
import sys
import torch
from torch.utils.data import DataLoader
from model import SimpleCNN

# Adjust the Python path to include the project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_cifar10  # Import the dataset loading function

# Load CIFAR-10 test dataset
_, testloader = load_cifar10()

# Initialize model and load saved state
model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Testing loop
def test_model():
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

if __name__ == "__main__":
    test_model()
