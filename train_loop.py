import torch
import math
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
N = 1000
x = (torch.rand(N) * 2 - 1) * math.pi
x = x.unsqueeze(1)
y = torch.sin(x) + 0.1 * torch.randn(N).unsqueeze(1)
print(x.shape, y.shape)

perm = torch.randperm(N)                  # shuffle before splitting
train_idx, val_idx = perm[:800], perm[800:]
x_train, y_train = x[train_idx], y[train_idx]
x_val,   y_val   = x[val_idx],   y[val_idx]
# Ideally want a model with MSE about 0.01 so lets take the border to be <=0.02

batch_size = 32
n_train = x_train.shape[0]
num_epochs = 100

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32,1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
#predicted = 0.51
predictions = model(x_train)
loss = loss_fn(predictions, y_train)
print("Starting loss: ", loss)

training_loss = []
validation_loss = []
for epoch in range(num_epochs):
    order = torch.randperm(n_train)

    for i in range(0, n_train, batch_size):
        idx = order[i : i+ batch_size]
        optimizer.zero_grad() 

        xb = x_train[idx]
        yb = y_train[idx]

        
        predictions = model(xb)

        
        loss = loss_fn(predictions,yb)
        
        loss.backward()
        optimizer.step()   

    with torch.no_grad():
        predictions = model(x_train)
        loss = loss_fn(predictions, y_train)
        print(f"Trianing Loss in Epoch: {epoch}: ", loss.item())
        training_loss.append(loss.item())

        predictions = model(x_val)
        loss = loss_fn(predictions, y_val)
        print(f"Validation Loss in Epoch: {epoch}: ", loss.item())
        validation_loss.append(loss.item())




plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()

print(loss)