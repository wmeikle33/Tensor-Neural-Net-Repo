import torch
import torch.nn as nn
import torch.optim as optim

# Data
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

# Model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Training loop
for epoch in range(5000):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"epoch={epoch}, loss={loss.item():.4f}")

# Predictions
with torch.no_grad():
    print(model(X))
