
import torch
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])])
train = datasets.ImageFolder('./cats_and_dogs_filtered/train', transform= transforms)
test = datasets.ImageFolder('./cats_and_dogs_filtered/validation', transform= transforms)
train_loader = DataLoader(train, batch_size= 128, shuffle= True)
test_loader = DataLoader(test, batch_size= 128)
class CNNWithoutDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                 nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.MaxPool2d(kernel_size=2, stride=2))
        self.classify_head = nn.Sequential(nn.Flatten(),
                                           nn.Linear(128 * 28 * 28, 512),
                                           nn.Linear(512, 2))

    def forward(self, x):
        return self.classify_head(self.net(x))


model_withoutdropout = CNNWithoutDropout()
model_withoutdropout.to('cuda')
class CNNWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), stride= (1, 1), padding=(1, 1)),
                                 nn.Conv2d(32, 64, kernel_size= (3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.MaxPool2d(kernel_size= 2, stride= 2, padding= 0),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.MaxPool2d(kernel_size=2, stride=2))
        self.classify_head = nn.Sequential(nn.Flatten(),
                                           nn.Linear(128 * 28 * 28, 512, bias= True),
                                           nn.Dropout(0.5),
                                           nn.Linear(512, 2, bias= True))
    def forward(self, x):
        return self.classify_head(self.net(x))

model_withdropout = CNNWithDropout()
model_withdropout.to('cuda')



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_withoutdropout.parameters(), lr= 0.001)



for epoch in range(10):
    model_withoutdropout.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model_withoutdropout(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch - {epoch}, loss - {running_loss}')

optimizer = torch.optim.SGD(model_withdropout.parameters(), lr= 0.001)
for epoch in range(10):
    model_withdropout.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model_withdropout(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch - {epoch}, loss = {running_loss}')


all_preds_withdropout, all_labels_withdropout = [], []
all_preds_withoutdropout, all_labels_withoutdropout = [], []
with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to('cuda'), target.to('cuda')
        o1 = model_withdropout(input)
        o2 = model_withoutdropout(input)
        val, ind1 = torch.max(o1, dim = 1)
        val, ind2 = torch.max(o2, dim = 1)
        all_preds_withdropout.extend(ind1.to('cpu'))
        all_preds_withoutdropout.extend(ind2.to('cpu'))
        all_labels_withdropout.extend(target.to('cpu'))
        all_labels_withoutdropout.extend(target.to('cpu'))
from sklearn.metrics import accuracy_score
print(accuracy_score(all_preds_withdropout, all_labels_withdropout))
print(accuracy_score(all_preds_withoutdropout, all_labels_withoutdropout))