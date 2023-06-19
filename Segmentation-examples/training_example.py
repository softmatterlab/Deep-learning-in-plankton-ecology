import sys

sys.path.append("..")

from unet_model import UNet
from torchinfo import summary

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Prepare data

training_set = torch.utils.data.TensorDataset(
    # images of shape (batch_size, channels, height, width),
    # labels of shape (batch_size, channels, height, width),
)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)


model = UNet(
    input_shape=(1, 1, 256, 256),
    number_of_output_channels=3,  # 2 for binary segmentation and 3 for multiclass segmentation
    conv_layer_dimensions=(8, 16, 32, 64, 128, 256),  # smaller UNet (faster training)
    # conv_layer_dimensions=(64, 128, 256, 512, 1024),  # standard UNet
)

summary(model, input_size=(1, 1, 256, 256), device=device)

model.to(device)


epochs = 100

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# looping over epochs
for epoch in range(epochs):

    model.train(True)

    # looping over batches
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(loss.item())
