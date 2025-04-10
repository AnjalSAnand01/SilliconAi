import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from google.colab import files
import math


def load_observed_corner_features(file_paths):
    """
    Takes list of file paths to CSVs of 5 observed corners.
    Returns: Tensor of shape (num_paths, 5, 3)
    """
    feature_list = []

    for file in file_paths:
        df = pd.read_csv(file)
        features = df[['slack', 'uncertainty', 'skew']].values  # shape: (num_paths, 3)
        feature_list.append(features)

    # Stack into shape: (num_paths, 5, 3)
    data = np.stack(feature_list, axis=1)
    return torch.tensor(data, dtype=torch.float32)

def load_target_slack(file_path):
    """
    Loads target slack values from a CSV (10 columns, 1 per corner).
    Returns: Tensor of shape (num_paths, 10)
    """
    df = pd.read_csv(file_path)
    return torch.tensor(df.values, dtype=torch.float32)


num_paths = 1000

# Generate 5 observed corners with 3 features
for i in range(5):
    df = pd.DataFrame({
        'path_id': np.arange(num_paths),
        'slack': np.random.normal(0, 0.05, num_paths),
        'uncertainty': np.random.uniform(0.01, 0.05, num_paths),
        'skew': np.random.normal(0.005, 0.002, num_paths),
    })
    df.to_csv(f'corner_obs_{i}.csv', index=False)

# Generate 10 target corners (just slack values)
target_df = pd.DataFrame(
    np.random.normal(0, 0.05, (num_paths, 10)),
    columns=[f"corner_{i}" for i in range(10)]
)
target_df.to_csv('corner_target.csv', index=False)

observed_paths = [f"corner_obs_{i}.csv" for i in range(5)]
X = load_observed_corner_features(observed_paths)  # shape: (1000, 5, 3)
Y = load_target_slack("corner_target.csv")    # shape: (1000, 10)

print("X shape:", X.shape)  # (1000, 5, 3)
print("Y shape:", Y.shape)  # (1000, 10)


class SlackPredictorWithEmbedding(nn.Module):
    def __init__(self):
        super(SlackPredictorWithEmbedding, self).__init__()
        
        self.corner_embed = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        # x: (batch_size, 5, 3)
        x = self.corner_embed(x)    # -> (batch_size, 5, 1)
        x = x.squeeze(-1)           # -> (batch_size, 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output_layer(x)
        return x


# Assuming X and Y are tensors (from the earlier loading function)
dataset = TensorDataset(X, Y)

# Create a DataLoader (for batching during training)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SlackPredictorWithEmbedding()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)  # Shape: (batch_size, 10)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print(model)

model.eval()
with torch.no_grad():
    test_preds = model(X[:5])
    print("Predicted slack values:\n", test_preds)
    print("True slack values:\n", Y[:5])
