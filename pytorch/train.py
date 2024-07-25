import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn

# Add the project root directory to the Python path
# This step is a solution to Python not being able to find the model / data modules during runtime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_cifar10  # Import the dataset loading function
from model import SimpleCNN

def train_model(start_epoch=0):
    print("Starting training...")
    trainloader, _ = load_cifar10()
    model = SimpleCNN()
    if start_epoch > 0:
        model.load_state_dict(torch.load('model.pth'))
        print("Loaded model from epoch", start_epoch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(start_epoch, start_epoch + 5):  # Training for additional 5 epochs
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} complete")

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved!")

if __name__ == "__main__":
    if os.path.exists('model.pth'):
        user_input = input("'model.pth' already exists. Do you want to delete it and retrain? (y/n): ")
        if user_input.lower() == 'y':
            os.remove('model.pth')
            print("'model.pth' deleted. Retraining the model...")
            train_model()
        else:
            start_epoch = int(input("Enter the epoch number to continue training from: "))
            train_model(start_epoch=start_epoch)
    else:
        train_model()
