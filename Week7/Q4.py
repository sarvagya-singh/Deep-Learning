import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])])
train = datasets.ImageFolder('./cats_and_dogs_filtered/train', transform= transforms)
test = datasets.ImageFolder('./cats_and_dogs_filtered/validation', transform= transforms)
train_loader = DataLoader(train, batch_size= 128, shuffle= True)
test_loader = DataLoader(test, batch_size= 128)


class dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return (x * mask) / (1 - self.p)
        return x


class CNN(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                 nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.MaxPool2d(kernel_size=2, stride=2))
        self.classify_head = nn.Sequential(nn.Flatten(),
                                           nn.Linear(128 * 28 * 28, 512),
                                           dropout(p),
                                           nn.Linear(512, 2))

    def forward(self, x):
        return self.classify_head(self.net(x))


model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
print(f'Epoch - {epoch}, loss = {running_loss}')


all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for input, target in test_loader:
        output = model(input)
        val, ind = torch.max(output, dim = 1)
        all_preds.extend(ind)
        all_labels.extend(target)
from sklearn.metrics import accuracy_score
print(accuracy_score(all_preds, all_labels))
