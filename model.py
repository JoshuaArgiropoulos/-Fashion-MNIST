import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)  # Dropout layer

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.dropout_fc = nn.Dropout(0.5)  # Dropout for fully connected layer
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 classes in Fashion MNIST

    def forward(self, x):
        # Apply convolutions, followed by max pooling and dropout
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers with ReLU, dropout, and output layer
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
