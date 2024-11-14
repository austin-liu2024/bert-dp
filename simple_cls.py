import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # First layer with 128 neurons
        self.fc2 = nn.Linear(256, output_dim)  # Output layer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)               # Output layer (logits)
        return x
