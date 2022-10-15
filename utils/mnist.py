import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100
num_epochs = 10

# Import MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

act_size = 64
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def train(self, criterion, optimizer):
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):  
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.forward(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


    def evaluate(self):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

    def adv_train(self, criterion, optimizer, attacker):
      n_total_steps = len(train_loader)
      for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):  
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)

                # Forward pass
                #outputs = self.forward(images)
                adv = attacker.perturb(images, labels)
                adv_outputs = self.forward(adv)
                loss = criterion(adv_outputs, labels)

                #loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Adv_Loss: {loss.item():.4f}')

    def adv_evaluate(self, attacker):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                adv = attacker.perturb(images, labels)
                adv_outputs = self.forward(adv)
                #outputs = self.forward(images)
                # max returns (value ,index)
                _, predicted = torch.max(adv_outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Adversary accuracy of the network on the 10000 test images: {acc} %')

class NeuralNetToy(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 16) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


class NeuralNetToy2(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 8) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(8, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class NeuralNet2(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 64) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class NeuralNet2_128(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 128) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class NeuralNet2_256(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 256) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class NeuralNet2_512(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 512) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class NeuralNet3(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, act_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(act_size, act_size)
        self.l3 = nn.Linear(act_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

class NeuralNet7(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, act_size) 
        self.l2 = nn.Linear(act_size, act_size)
        self.l3 = nn.Linear(act_size, act_size)
        self.l4 = nn.Linear(act_size, act_size)
        self.l5 = nn.Linear(act_size, act_size)
        self.l6 = nn.Linear(act_size, act_size)
        self.l7 = nn.Linear(act_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.l6(out)
        out = self.relu(out)
        out = self.l7(out)
        # no activation and no softmax at the end
        return out

class NeuralNet8(NeuralNet):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, act_size) 
        self.l2 = nn.Linear(act_size, act_size)
        self.l3 = nn.Linear(act_size, act_size)
        self.l4 = nn.Linear(act_size, act_size)
        self.l5 = nn.Linear(act_size, act_size)
        self.l6 = nn.Linear(act_size, act_size)
        self.l7 = nn.Linear(act_size, act_size)
        self.l8 = nn.Linear(act_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.l6(out)
        out = self.relu(out)
        out = self.l7(out)
        out = self.relu(out)
        out = self.l8(out)
        # no activation and no softmax at the end
        return out
