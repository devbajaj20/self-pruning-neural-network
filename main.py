import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os

# ================================
# 1. PRUNABLE LINEAR LAYER
# ================================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # IMPORTANT FIX: start near zero (encourage pruning)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) - 2.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


# ================================
# 2. MODEL
# ================================
class PrunableNet(nn.Module):
    def __init__(self):
        super(PrunableNet, self).__init__()

        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ================================
# 3. DATASET
# ================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ================================
# 4. SETUP
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

os.makedirs("results", exist_ok=True)

print("Setup complete. Ready for training...")


# ================================
# 5. SPARSITY LOSS (FIXED)
# ================================
def compute_sparsity_loss(model):
    loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.mean(gates)   # FIX: mean instead of sum
    return loss


# ================================
# 6. TRAINING
# ================================
def train(model, train_loader, optimizer, lambda_val, epochs=5):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            ce_loss = criterion(output, target)
            sparsity_loss = compute_sparsity_loss(model)

            # FIX: scale sparsity loss
            loss = ce_loss + lambda_val * sparsity_loss * 100

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# ================================
# 7. EVALUATION
# ================================
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicted = torch.max(output, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100 * correct / total


def compute_sparsity(model, threshold = 0.1):
    total_weights = 0
    pruned_weights = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)

            total_weights += gates.numel()
            pruned_weights += torch.sum(gates < threshold).item()

    return 100 * pruned_weights / total_weights


# ================================
# 8. PLOT GATE DISTRIBUTION
# ================================
def plot_gate_distribution(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")
    plt.savefig("results/gate_distribution.png")
    plt.show()


# ================================
# 9. RUN EXPERIMENT
# ================================
lambda_values = [0.01, 0.1, 1.0]

results = []

best_model = None
best_lambda = None
best_sparsity = 0

for lambda_val in lambda_values:
    print(f"\nTraining with lambda = {lambda_val}")

    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, optimizer, lambda_val, epochs=5)

    accuracy = evaluate(model, test_loader)
    sparsity = compute_sparsity(model)

    print(f"Lambda: {lambda_val}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    results.append((lambda_val, accuracy, sparsity))

    # pick best (high sparsity but reasonable accuracy)
    if sparsity > best_sparsity and accuracy >= 50:
        best_sparsity = sparsity
        best_model = model
        best_lambda = lambda_val

print("\nFinal Results:")
for r in results:
    print(f"Lambda={r[0]}, Acc={r[1]:.2f}%, Sparsity={r[2]:.2f}%")

# ================================
# SAVE RESULTS TO FILE
# ================================
with open("results/results.txt", "w") as f:
    f.write("Lambda\tAccuracy\tSparsity\n")
    for r in results:
        f.write(f"{r[0]}\t{r[1]:.2f}\t{r[2]:.2f}\n")

print("Results saved to results/results.txt")

print(f"\nBest Model Lambda: {best_lambda}")

# Plot distribution
plot_gate_distribution(best_model)

# Save best model
torch.save(best_model.state_dict(), "results/best_model.pth")

print("Best model saved at results/best_model.pth")