import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Functions to read the IDX files
def read_images(filename):
    with open(filename, 'rb') as f:
        # Read and verify the magic number
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} in image file {filename}')
        # Read dimensions
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)
    return data

def read_labels(filename):
    with open(filename, 'rb') as f:
        # Read and verify the magic number
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} in label file {filename}')
        # Read number of labels
        num_labels = struct.unpack('>I', f.read(4))[0]
        # Read label data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.shape[0] != num_labels:
            raise ValueError('Mismatch in label count')
    return data

# Preprocessing function
def preprocess_images(images):
    # Pad images to 32x32
    images_padded = np.pad(images, ((0, 0), (2, 2), (2, 2)), 'constant')
    # Normalize images to [0,1]
    images_normalized = images_padded.astype(np.float32) / 255.0
    # Add channel dimension
    images_normalized = np.expand_dims(images_normalized, axis=1)  # (N, 1, 32, 32)
    return images_normalized

# Load and preprocess data
train_images = read_images('train-images.idx3-ubyte')
train_labels = read_labels('train-labels.idx1-ubyte')
test_images = read_images('t10k-images.idx3-ubyte')
test_labels = read_labels('t10k-labels.idx1-ubyte')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# Convert to PyTorch tensors
train_images = torch.from_numpy(train_images)
train_labels = torch.from_numpy(train_labels).long()
test_images = torch.from_numpy(test_images)
test_labels = torch.from_numpy(test_labels).long()

# Create datasets and loaders
batch_size = 64
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the LeNet-5 model with ReLU activations
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        # No activation at the output layer

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Raw logits

# Initialize the model, loss function, optimizer, and device
model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer
device = torch.device("cpu")  # Use CPU only
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # Raw logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # Evaluation on test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)  # Raw logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {100 * correct / total:.2f}%')

# Inference function
def predict(model, image):
    # image: (1, 1, 32, 32) tensor
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)  # Raw logits
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Example inference on a test image
test_image = test_images[0].unsqueeze(0)  # (1, 1, 32, 32)
predicted_label = predict(model, test_image)
print(f'Predicted Label: {predicted_label}, True Label: {test_labels[0].item()}')