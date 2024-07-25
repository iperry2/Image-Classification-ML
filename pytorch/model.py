import torch.nn as nn
import torch.nn.functional as F

# Class that defines simple CNN model
# initializes PyTorch module
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)           # 1st Convolutional Layer. Input: 3 channels (RGB), Output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)          # 2nd Convolutional Layer. Input: 32 channels, Output: 64
        self.fc1 = nn.Linear(64 * 8 * 8, 128)                                       # 1st fully connected Layer. Input: 64 * 8 * 8, Output: 128                                      
        self.fc2 = nn.Linear(128, 10)                                               # 2nd fully connected Layer. Input: 128, Output: 10

    def forward(self, x):
        x = F.relu(self.conv1(x))           # Applies 1st Convolutional Layer and ReLU activation function
        x = F.max_pool2d(x, 2)              # Applies 2x2 max pooling, reduces spatial dimensions by half
        x = F.relu(self.conv2(x))           # Applies the second convolutional layer followed by another ReLU activation.
        x = F.max_pool2d(x, 2)              # Another max pooling operation
        x = x.view(-1, 64 * 8 * 8)          # Flattens the output of the convolution
        x = F.relu(self.fc1(x))             # Applies the first fully connected layer followed by ReLU activation
        x = self.fc2(x)                     # Applies the second fully connected layer
        return x
