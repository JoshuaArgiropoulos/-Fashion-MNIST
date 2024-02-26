import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torchvision

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Mapping of Fashion MNIST classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def show_images(images, labels, classes):
    images = torchvision.utils.make_grid(images[:4])
    np_images = images.numpy()
    plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    plt.title('Sample Images')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('Labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


def imshow(img):
    """
    Show an image from a torch tensor.
    """
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_batch(images, labels):
    """
    Display a batch of images and labels.
    """
    imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(' '.join(f'{classes[labels[j]]}' for j in range(len(labels))))

def plot_training_history(history):
    """
    Plot training loss and accuracy history.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def show_predictions(images, labels, preds, classes):
    plt.figure(figsize=(10,4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        img = images[i] / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f'Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}')
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_first_layer_filters(model):
    with torch.no_grad():
        weights = model.conv1.weight.data.cpu()
        fig, axes = plt.subplots(1, weights.size(0))
        for i, ax in enumerate(axes):
            ax.imshow(weights[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.show()



def plot_roc_curve(y_test, y_score, classes):
    # Binarize the labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=list(range(len(classes))))
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.show()