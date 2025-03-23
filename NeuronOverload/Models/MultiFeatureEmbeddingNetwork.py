#4  feature per Path-Corner
#initial embedding layer in network will embedd 3 feature value to one inpu
#8 to 14 corner slack prediction


import torch
import torch.nn as nn
import torch.optim as optim

class PathCornerModel(nn.Module):
    def __init__(self, num_corners_observed=8, num_corners_unobserved=14, embedding_dim=1, hidden_dim=32):
        super(PathCornerModel, self).__init__()
        self.embedding = nn.Linear(3, embedding_dim)
        self.fc1 = nn.Linear(num_corners_observed * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_corners_unobserved)
    
    def forward(self, x):
        x = x.view(-1, 8, 3)
        x = self.embedding(x)
        x = x.view(-1, 8 * 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.output(x)
        return output

num_samples = 1000
num_observed_corners = 8
num_unobserved_corners = 14

X_train = torch.randn(num_samples, num_observed_corners * 3)
y_train = torch.randn(num_samples, num_unobserved_corners)

model = PathCornerModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")