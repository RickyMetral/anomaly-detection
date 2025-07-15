import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import random
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize images to 224x224
    transforms.ToTensor(), # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize images
])


train_data = datasets.ImageFolder(root = train_path, transform=transform)
test_data  = datasets.ImageFolder(root = test_path, transform=transform)

loaders = {
    "train" : DataLoader(train_data,
                        batch_size = 32,
                        shuffle = True,
                        num_workers = 1),
    "test"  : DataLoader(test_data,
                        batch_size = 32,
                        shuffle = True,
                        num_workers = 1)
}
class AnomalyDetection(nn.Module):
    max_channels = 20
    def __init__(self, in_channels, max_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, max_channels, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.glob_avg_pool = nn.AvgPool2d()
        self.linear1 = nn.Linear(max_channels, 1)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.glob_avg_pool(x, self.max_channels))
        x = F.relu(self.linear1(x))

        return F.Sigmoid(x)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_large(weights="DEFAULT")

for param in model.parameters():
    param.requires_grad = False

model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
loss_fn = nn.BCELoss()

model.classifier[3] = nn.Linear(1280, 1)
model.classifier.add_module("4", nn.Sigmoid())

def train(model, epoch):
    model.train()
    for num_batch, (data, label) in enumerate(loaders["train"]):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        if num_batch % 10 == 0:
            print(f"Train epoch: {epoch} [{num_batch * len(data)}/{len(loaders['train'].dataset)} ({100. * num_batch / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")


def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target.unsqueeze(1).float()).item() * len(data) # Multiply by batch size to get total loss for the batch
            pred = (output >= 0.5).squeeze() # Remove the extra dimension
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss/=len(loaders["test"].dataset)
    print(f"\nTest Set: Average Loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders['test'].dataset)} ({100* correct / len(loaders['test'].dataset):.0f}%\n)")

for epoch in range(1, 5):
    train(model, epoch)
    test(model)


model.eval()
preds = ['Anomaly', "Normal"]

# Get a random sample from the dataset
index = random.randint(0, len(test_data) - 1)
data, target = test_data[index]


# Add batch dimension and move to device
data = data.unsqueeze(0).to(device)

# Inference
output = model(data)
pred = (target >= 0.5)
print(f"Label: {preds[target]}")

# Convert the image tensor for display
# Assuming the tensor shape is [1, 28, 28] (grayscale), we squeeze and use matplotlib
image = data.squeeze(0).permute(1,2,0).cpu().numpy()

# Plotting the image
plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {preds[pred]}")
plt.axis('off')
plt.show()