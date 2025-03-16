import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ===================== Utility Functions ===================== #

def relu(x):
    

def softmax(x):
    


# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

# ===================== CNN Structure ===================== #
class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        

    def forward(self, x):
        """ Forward propagation """
        

        return outputs

    def backward(self, x, y, pred):
        """ Backward propagation """
        # 1. one-hot encode the labels

        # 2. Calculate softmax cross-entropy loss gradient
        
        # 3. Calculate fully connected layer gradient
        
        # 4. Backpropagate through ReLU
        
        # 5. Calculate convolution kernel gradient
        
        # 6. Update parameters
        

    def train(self, x, y):
        # call forward function
        
        # calculate loss
        
        # call backward function

        return loss

# ===================== Training Process ===================== #
def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28
    num_epochs = 5
    

    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, lables in train_loader:  # define training phase for training model
            

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")

if __name__ == "__main__":
    main()
