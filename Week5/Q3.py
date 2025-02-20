import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self._to_linear = None
        self._calculate_conv_output_size((1, 28, 28))

        self.classification_head = nn.Sequential(
            nn.Linear(self._to_linear, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 10, bias=True)
        )

    def _calculate_conv_output_size(self, input_size):
        with torch.no_grad():
            x = torch.ones(1, *input_size)
            x = self.net(x)
            self._to_linear = x.numel()

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=50, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=50, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

correct = np.sum(np.array(all_preds) == np.array(all_labels))
total = len(all_labels)
accuracy = correct/total
print(f'Accuracy: {accuracy * 100:.4f}%')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f'Number of learnable parameters: {num_params}')

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Output :
    # Epoch [1/5], Loss: 0.3153394287040768
    # Epoch [2/5], Loss: 0.09279175668217553
    # Epoch [3/5], Loss: 0.06799774197250372
    # Epoch [4/5], Loss: 0.052092426602515236
    # Epoch [5/5], Loss: 0.0430296938434185
    # Accuracy: 98.2000%
    # Number of learnable parameters: 149798
