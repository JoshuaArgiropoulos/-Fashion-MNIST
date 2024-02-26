import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from model import FashionMNISTCNN
from data_loader import get_data_loaders  # Import the get_data_loaders function
from utils import show_images
from utils import plot_training_history
from utils import classes

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize data loaders
train_loader, test_loader = get_data_loaders(train_batch_size=64, test_batch_size=1000)

images, labels = next(iter(train_loader))
show_images(images, labels, classes)

model = FashionMNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
history = {
    'train_loss': [],
    'train_acc': [], 
    'val_loss': [], 
    'val_acc': [] 
}
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    total_correct = 0  # For accuracy calculation
    total_images = 0  # For accuracy calculation

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_images += labels.size(0)

    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:  
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_avg_loss = val_loss / len(test_loader)
    val_avg_acc = val_correct / val_total
    

   
    avg_loss = total_loss / len(train_loader)

    avg_acc = total_correct / total_images  # Calculate average accuracy
    val_avg_acc = val_correct / val_total

     # Update history
    history['train_loss'].append(avg_loss)
    # For accuracy
    history['train_acc'] = history.get('train_acc', []) + [avg_acc]  
    #Update Val 
    history['val_loss'].append(val_avg_loss)
    history['val_acc'].append(val_avg_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Val Acc: {val_avg_acc:.4f}")

    

# After the training loop, plot the training history
plot_training_history(history)



torch.save(model.state_dict(), 'models/fashion_mnist_cnn.pth')