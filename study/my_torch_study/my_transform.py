import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import os
from torch import nn
from torchvision import datasets , transforms
from torch.utils.data import DataLoader

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download=True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# to tensor

#lambda transform
target_transform = Lambda(lambda Y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index = torch.tensor(y), value =1))


def test_network():
############build the neural network##########################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} device'.format(device))

#define the class
    class NeuralNetWork(nn.Module):
        def __init__(self):
            super(NeuralNetWork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetWork().to(device)
    print(model)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probad = nn.Softmax(dim=1)(logits)
    y_pred = pred_probad.argmax(1)
    print(f"Predicted class:{y_pred}")



#test model layers
def test_model_layers():
    input_image = torch.randn(3, 28, 28)
    print(input_image.size())

    #nn.flatten
    #We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values ( the minibatch dimension (at dim=0) is maintained).
    flatten = nn.Flatten()
    flatten_image = flatten(input_image)
    print(flatten_image.size())

    #nn.linear
    #The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
    layer1 = nn.Linear(in_features=28*28, out_features=28)
    hidden1 = layer1(flatten_image)
    print(hidden1.size())

    #nn.relu
    #Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity,
    # helping neural networks learn a wide variety of phenomena.
    print(f"Before Relu: :{hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After relu {hidden1}\n\n")

    #nn.Sequential
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU,
        nn.Linear(20, 10)
    )
    input_image = torch.randn(3, 28, 28)
    logits = seq_modules(input_image)

    #nn.softmax
    softmax = nn.Softmax(dim = 1)
    pred_probab = softmax(logits)

    #model parameters
    #In this example, we iterate over each parameter, and print its size and a preview of its values.
    print("model structure ", seq_modules,  "\n\n")
    for name, param , in seq_modules.named_parameters():
        print(f"layer {name},size: {param.size()}, value: {param[:2]}\n")





