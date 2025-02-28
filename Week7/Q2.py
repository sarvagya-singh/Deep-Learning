import torch
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])])
train = datasets.ImageFolder('./cats_and_dogs_filtered/train', transform= transforms)
test = datasets.ImageFolder('./cats_and_dogs_filtered/validation', transform= transforms)
train_loader = DataLoader(train, batch_size= 128, shuffle= True)
test_loader = DataLoader(test, batch_size= 128)


model = models.alexnet(weights= 'IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
print(model)

model.classifier[6] = nn.Linear(4096, 2)
print(model)
model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.03)
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        l1 = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1 += torch.norm(param, p= 1)
        loss += 0.01 * l1
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch - {epoch+1}, loss = {running_loss}')
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to('cuda'), target.to('cuda')
            output = model(input)
            val, ind = torch.max(output, dim=1)
            all_preds.extend(ind.to('cpu'))
            all_labels.extend(target.to('cpu'))
    from sklearn.metrics import accuracy_score

    print(accuracy_score(all_preds, all_labels))