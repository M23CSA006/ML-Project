
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the USPS dataset
trainset = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.USPS(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(16 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 16 * 16)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the models
mlp_model = MLP()
cnn_model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Define TensorBoard writers
mlp_writer = SummaryWriter('./logs/mlp')
cnn_writer = SummaryWriter('./logs/cnn')

# Training loop for MLP
for epoch in range(10):  # 10 epochs as an example
    mlp_model.train()
    for inputs, labels in trainloader:
        mlp_optimizer.zero_grad()
        outputs = mlp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        mlp_optimizer.step()

    # Calculate metrics and write to TensorBoard
    mlp_model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        all_labels = []
        all_predicted = []
        for inputs, labels in testloader:
            outputs = mlp_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_predicted.extend(predicted.numpy())

        accuracy = correct / total
        precision = precision_score(all_labels, all_predicted, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predicted, average='macro', zero_division=0)
        confusion = confusion_matrix(all_labels, all_predicted)

        mlp_writer.add_scalar('Loss', loss, epoch)
        mlp_writer.add_scalar('Accuracy', accuracy, epoch)
        mlp_writer.add_scalar('Precision', precision, epoch)
        mlp_writer.add_scalar('Recall', recall, epoch)

        # Reshape confusion matrix tensor to CHW format for TensorBoard
        confusion_img = torch.tensor(confusion).unsqueeze(0)
        confusion_img = confusion_img.float() / confusion_img.max()  # Normalize the values

        # Add the confusion matrix image to TensorBoard
        mlp_writer.add_image('Confusion Matrix', confusion_img, epoch)

# Training loop for CNN (similar to MLP training loop, but using cnn_model)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir='./logs'