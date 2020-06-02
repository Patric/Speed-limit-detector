import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        # calling super class constructor to use parent methods later
        super(Net, self).__init__()
        #
        # first convolution layer
        # 1 - in channels, 6 - out -channels, 3x3 - kernel size
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation y = Wx + b
        # 6*6 from image dimension
        # in_features - size of each features sample
        # out_features = sieze of an output feature - image
       
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #   max pooling over a (2,2) window - take a maximum  valuefrom each square that window
        #   consists of
        #   relu applies function to each element of the matrix
        #   rectifier - activation function defined as the positive part of its argument
        #   also called ramp function. Rectifier - prostownik
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # batch size - numer of samples that will be propagated through the network
        # all dimension except the batch dimension
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
# print(net)
params = list(net.parameters())
# print(len(params))
# print(params[0].size())

input = torch.randn(1, 1, 32, 32)

out = net(input)
# print(out)

net.zero_grad()
out.backward(torch.randn(1,10))


# Loss function
# A loss function takes the (output, target) pair of inputs, and computes a 
# value that estimates how far away the output is from the target.

# There are several different loss functions. A simple loss is nn.MSELoss which
# computes the mean-squared error between the input and the target

output = net(input)
target = torch.randn(10) # dummy target for example
target = target.view(1, -1) # make it the same shape as output

criterion = nn.MSELoss()

# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#     -> view -> linear -> relu -> linear -> relu -> linear
#     -> MSELoss
#     -> loss

loss = criterion(output, target)
# print(loss)

# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad() #zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


#The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):

#    weight = weight - learning_rate * gradient

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# However, as you use neural networks, you want to use various different 
# update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc. To enable this,
# we built a small package: torch.optim that implements all these methods. 
# Using it is very simple:

import torch.optim as optim
#create opmitimser
optimiser = optim.SGD(net.parameters(), lr = 0.01)

# in training loop"
optimiser.zero_grad() # zero the gradient buffers. They had to be set manually to 0,
# since they are accumulataed
output = net(input)
loss = criterion(output, target)
loss.backward()
optimiser.step() #updates