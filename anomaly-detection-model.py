import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

train_data = datasets.
test_data  = datasets.
in_features

loaders = {
    "train" : DataLoader(train_data,
                        batch_size = ___,
                        shuffle = True,
                        num_workers = ___),
    "test"  : DataLoader(test_data,
                        batch_size = 20,
                        shuffle = True,
                        num_workers = ____)
}

class AnomalyDetection(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x  = F.max_pool2d(x, 2)


        return F.Sigmoid(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnomalyDetection().to(device)
optimizer = optim.Adam(model.parameters(), lr = ____)
loss_fn = nn.BCEWithLogitsLoss()


def train(epoch):
    model.train()
    for num_batch, (data, label) in enumerate(loaders["train"]):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        if num_batch % 20 == 0:
            print(f"Train epoch: {epoch} [{num_batch * len(data)}/{len(loaders['train'].dataset)} ({100. * num_batch / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")




def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = loss_fn(output, target).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss/=len(loaders["test"].dataset)
    print(f"\nTest Set: Average Loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders['test'].dataset)} ({100* correct / len(loaders['test'].dataset):.0f}%\n)")


