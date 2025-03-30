import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the LeNet-5 model with Dropout
class LeNet5Dropout(nn.Module):
    def __init__(self):
        super(LeNet5Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 120)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define the LeNet-5 model with Weight Decay
class LeNet5WeightDecay(nn.Module):
    def __init__(self):
        super(LeNet5WeightDecay, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the LeNet-5 model with Batch Normalization
class LeNet5BatchNorm(nn.Module):
    def __init__(self):
        super(LeNet5BatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)
        self.bn3 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the LeNet-5 model without Regularization
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training and evaluation function
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

    return train_accuracies, test_accuracies

# Plot the convergence graph
def plot_convergence(train_accuracies, test_accuracies, title):
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'b', label='Train accuracy')
    plt.plot(epochs, test_accuracies, 'r', label='Test accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Plot the results table
def plot_results_table(results):
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table_data = [["Model", "Train Accuracy", "Test Accuracy"]] + results
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)
    plt.show()

# Example usage
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the networks
net_dropout = LeNet5Dropout()
net_weight_decay = LeNet5WeightDecay()
net_batch_norm = LeNet5BatchNorm()
net_no_reg = LeNet5()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizers
optimizer_dropout = optim.SGD(net_dropout.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_weight_decay = optim.SGD(net_weight_decay.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_batch_norm = optim.SGD(net_batch_norm.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_no_reg = optim.SGD(net_no_reg.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

# Train and evaluate the models
train_accuracies_dropout, test_accuracies_dropout = train_and_evaluate(net_dropout, train_loader, test_loader, criterion, optimizer_dropout, num_epochs=10)
train_accuracies_weight_decay, test_accuracies_weight_decay = train_and_evaluate(net_weight_decay, train_loader, test_loader, criterion, optimizer_weight_decay, num_epochs=10)
train_accuracies_batch_norm, test_accuracies_batch_norm = train_and_evaluate(net_batch_norm, train_loader, test_loader, criterion, optimizer_batch_norm, num_epochs=10)
train_accuracies_no_reg, test_accuracies_no_reg = train_and_evaluate(net_no_reg, train_loader, test_loader, criterion, optimizer_no_reg, num_epochs=10)

# Plot the convergence graphs
plot_convergence(train_accuracies_dropout, test_accuracies_dropout, 'LeNet-5 with Dropout')
plot_convergence(train_accuracies_weight_decay, test_accuracies_weight_decay, 'LeNet-5 with Weight Decay')
plot_convergence(train_accuracies_batch_norm, test_accuracies_batch_norm, 'LeNet-5 with Batch Normalization')
plot_convergence(train_accuracies_no_reg, test_accuracies_no_reg, 'LeNet-5 without Regularization')

# Collect results
results = [
    ["LeNet-5 with Dropout", train_accuracies_dropout[-1], test_accuracies_dropout[-1]],
    ["LeNet-5 with Weight Decay", train_accuracies_weight_decay[-1], test_accuracies_weight_decay[-1]],
    ["LeNet-5 with Batch Norm", train_accuracies_batch_norm[-1], test_accuracies_batch_norm[-1]],
    ["LeNet-5 without Reg", train_accuracies_no_reg[-1], test_accuracies_no_reg[-1]]
]

# Plot the results table
plot_results_table(results)

# Determine the winner
best_model = max(results, key=lambda x: x[2])
print(f"The best model is {best_model[0]} with a test accuracy of {best_model[2]:.2f}%")