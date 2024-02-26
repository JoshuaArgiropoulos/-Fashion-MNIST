import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from model import FashionMNISTCNN
from data_loader import get_data_loaders  # Import the get_data_loaders function

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize data loaders
train_loader, test_loader = get_data_loaders(train_batch_size=64, test_batch_size=1000)

model = FashionMNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients for this training step
        outputs = model(images)  # Forward pass: Compute predicted outputs by passing inputs to the model
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'models/fashion_mnist_cnn.pth')