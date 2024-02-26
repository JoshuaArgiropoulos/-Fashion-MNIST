import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_batch_size, test_batch_size):
     # Transformations for the training set
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the images on the horizontal
        transforms.RandomRotation(10),      # Randomly rotate the images by 10 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        # Add more transforms as needed
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=train_transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # Loading the test set
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=test_transforms, download=True)
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
