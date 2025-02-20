import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(64, 128, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(128, 64, kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2))
        self.classify_head = nn.Sequential(nn.Linear(64, 20, bias=True),
                                           nn.ReLU(),
                                           nn.Linear(20, 10, bias=True))
    def forward(self, x):
        features = self.net(x)
        features = features.view(features.size(0), -1)
        return self.classify_head(features)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def save_checkpoint(model, optimizer, epoch, loss, best_model_path="best_model.pth", checkpoint_path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}.")

    torch.save(model.state_dict(), best_model_path)
    print(f"Best model saved at epoch {epoch}.")


def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: Resume from epoch {epoch}, Loss: {loss}")
    return epoch, loss


def train(model, train_loader, criterion, optimizer, epochs=1000, checkpoint_path="checkpoint.pth"):
    best_loss = float('inf')
    start_epoch = 0

    try:
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    model.train()
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_path=checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch + 1, avg_loss)

def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm


train(model, train_loader, criterion, optimizer, epochs=1000)

cm = evaluate(model, test_loader)
print(cm)
print(f"Number of learnable parameters: {count_parameters(model)}")