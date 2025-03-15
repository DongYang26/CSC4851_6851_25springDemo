"""
MNIST Dataset Loading Example for Deep Learning Projects
=====================================================

This script demonstrates how to load and preprocess the MNIST dataset using PyTorch.
MNIST is a database of handwritten digits (0-9) commonly used for training various
image processing and machine learning systems.

Key Features of MNIST:
- Training Set: 60,000 examples
- Test Set: 10,000 examples
- Image Size: 28x28 pixels (grayscale)
- Labels: Digits from 0 to 9

Example Usage in Your Project:
-----------------------------
from read_MNIST import load_data

# Get train and test dataloaders
train_loader, test_loader = load_data()

# Iterate through batches
for images, labels in train_loader:
    # images shape: [batch_size, 1, 28, 28]
    # labels shape: [batch_size]
    
    # Your training code here
    # Example:
    # forward propagation based on input images to get outputs
    # calculate loss between outputs and labels by loss function you designed
    # backward propagation based on both backward function you designed and calculated loss
    
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def dataloader(train_dataset, test_dataset, batch_size=128):
    """
    Creates DataLoader objects for both training and testing datasets.
    
    Parameters:
    -----------
    train_dataset : torch.utils.data.Dataset
        The training dataset for training model
    test_dataset : torch.utils.data.Dataset
        The testing dataset for testing model
    batch_size : int, optional (default=128)
        Number of samples per batch
        
    Returns:
    --------
    tuple(DataLoader, DataLoader)
        Train and test dataloaders
        
    Example:
    --------
    # If you want to use a different batch size:
    train_loader, test_loader = dataloader(train_dataset, test_dataset, batch_size=64)
    """
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True)  # shuffle=True for random batch sampling
    
    test_loader = DataLoader(dataset=test_dataset, 
                           batch_size=batch_size, 
                           shuffle=False)  # shuffle=False for consistent testing
    
    return train_loader, test_loader


def load_data():
    """
    Using this function to loads and preprocesses the MNIST dataset !!! ^_^
    
    The preprocessing steps include:
    1. Converting images to PyTorch tensors
    2. Normalizing pixel values to [-1, 1] range
       Original pixel values are [0, 255], after ToTensor() they become [0, 1]
       Normalize((0.5,), (0.5,)) transforms them to [-1, 1]
    
    Returns:
    --------
    tuple(DataLoader, DataLoader)
        Train and test dataloaders for training phase and testing phase respectively
        
    Example Usage:
    -------------
    # Basic usage
    train_loader, test_loader = load_data()
    
    # Accessing a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")  # Expected: [128, 1, 28, 28]
        print(f"Labels shape: {labels.shape}")  # Expected: [128]
        break  # Just showing the first batch
        
    # Common training loop structure:
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 1. Forward pass
            
            # 2. Calculate loss
            
            # 3. Backward pass
            
    """
    # Define the preprocessing transformations
    transform = transforms.Compose([transforms.ToTensor(),  # Convert images to tensor and scale to [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",  # Data will be downloaded here
        train=True,  # Specify training dataset
        download=True,  # Download if not already present
        transform=transform  # Apply the preprocessing
    )
    
    # Load testing data
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=False,  # Specify test dataset
        download=True,
        transform=transform
    )

    print("The number of training data:", len(train_dataset))  # Should print 60000
    print("The number of testing data:", len(test_dataset))   # Should print 10000

    return dataloader(train_dataset, test_dataset)  # using above designed dataloader() function for here
