from reader import *
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def preprocess(A):
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = []
    for node_adjaceny in A:
        num = 0
        for node in node_adjaceny:
            if node == 1.0:
                num = num + 1
        # Add an extra for the "self loop"
        num = num + 1
        degrees.append(num)
    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(degrees)
    # Cholesky decomposition of D
    D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Turn adjacency matrix into a numpy matrix
    A = np.matrix(A)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    return A_hat, D


x , y = read_data('karate.data', 'label.data')
A , D = preprocess(x)
print A
print D

FloatTensor = torch.FloatTensor

# Turn the input and output into FloatTensors for the Neural Network
x = Variable(FloatTensor(x), requires_grad=False)
y = Variable(FloatTensor(y), requires_grad=False)
A = torch.from_numpy(A)
A = A.float()
A = Variable(A, requires_grad=False)
D = torch.from_numpy(D)
D = D.float()
D = Variable(D, requires_grad=False)

# Create random tensor weights
W1 = Variable(torch.randn(34, 34).type(FloatTensor), requires_grad=True)
W2 = Variable(torch.randn(34, 2).type(FloatTensor), requires_grad=True)
W3 = Variable(torch.randn(2, 1).type(FloatTensor), requires_grad=True)

learning_rate = 1e-6
for t in range(5000):

    hidden_layer_1 = F.relu(D.mm(A).mm(D).mm(x).mm(W1))
    hidden_layer_2 = F.relu(D.mm(A).mm(D).mm(hidden_layer_1).mm(W2))
    y_pred = F.relu(D.mm(A).mm(D).mm(hidden_layer_2).mm(W3))

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    loss.backward()

    # Update weights using gradient descent
    W1.data -= learning_rate * W1.grad.data
    W2.data -= learning_rate * W2.grad.data
    W3.data -= learning_rate * W3.grad.data

    # Manually zero the gradients after updating weights
    W1.grad.data.zero_()
    W2.grad.data.zero_()
    W3.grad.data.zero_()