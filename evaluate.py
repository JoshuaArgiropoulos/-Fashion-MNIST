import torch
from model import FashionMNISTCNN
from data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_, test_loader = get_data_loaders(train_batch_size=64, test_batch_size=1000)

model = FashionMNISTCNN().to(device)
model.load_state_dict(torch.load('models/fashion_mnist_cnn.pth'))
model.eval()  # Set the model to evaluation mode

correct = 0
total = 0
with torch.no_grad():  # No need to track gradients
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total}%')
