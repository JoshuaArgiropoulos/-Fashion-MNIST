import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # Loading the test set
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(train_batch_size=64, test_batch_size=1000)

    # Example: Iterate over the train_loader
    for images, labels in train_loader:
        # Here, images and labels are batches of images and labels, respectively.
        # You can now use these batches to train your model.
        print(images.shape, labels.shape)
