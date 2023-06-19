#%%
import sys

sys.path.append("..")

import os
import numpy as np
from tqdm import tqdm
import torch
from torchinfo import summary
import tensorflow as tf
import matplotlib.pyplot as plt

from pytorch.unet_model import UNet
from pytorch.losses import CustomBCEloss

from simulation.noctiluca import generate_images

# from simulation.fluorescence_particles import generate_images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def batch_function(x):
    return x  # / np.median(x)


def transform_masks(m1, m2):
    combined_mask = m1 + m2 * 2
    combined_mask[combined_mask == 3] = 1  # When overlapping, only keep the first mask
    return combined_mask


# Normalise between 0 and 1
# def batch_function(x):
#     return (x - np.min(x)) / (np.max(x) - np.min(x))


# Generate training data
train_images = []
train_labels = []
for i in tqdm(range(1024), desc="Generating training data"):
    img, mask1, mask2 = generate_images().update()()
    img = batch_function(np.array(img))
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    train_images.append(img)
    train_labels.append(transform_masks(mask1, mask2))  # already summed
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Generate validation data
val_images = []
val_labels = []
for i in tqdm(range(256), desc="Generating validation data"):
    img, mask1, mask2 = generate_images().update()()
    img = batch_function(np.array(img))
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    val_images.append(img)
    val_labels.append(transform_masks(mask1, mask2))  # already summed
val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Prepare training data and loaders
training_set = torch.utils.data.TensorDataset(
    torch.from_numpy(train_images).float().permute(0, 3, 1, 2),
    torch.from_numpy(train_labels).float().permute(0, 3, 1, 2),
)
trainloader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)

# Create validation data and loaders
validation_set = torch.utils.data.TensorDataset(
    torch.from_numpy(val_images).float().permute(0, 3, 1, 2),
    torch.from_numpy(val_labels).float().permute(0, 3, 1, 2),
)
validationloader = torch.utils.data.DataLoader(
    validation_set, batch_size=32, shuffle=True
)

# Declare the model and place it on the GPU
model = UNet(
    input_shape=(1, 1, 256, 256),
    number_of_output_channels=3,  # 2 for binary segmentation and 3 for multiclass segmentation
    conv_layer_dimensions=(8, 16, 32, 64, 128, 256),  # smaller UNet (faster training)
    # conv_layer_dimensions=(64, 128, 256, 512, 1024),  # standard UNet
)

# Print model summary
summary(model, input_size=(1, 1, 256, 256), device=device)

model.to(device)

# loss function and optimizer
# criterion = torch.nn.BCEWithLogitsLoss()
# criterion = CustomBCEloss()
# criterion = weighted_channel_BCEloss(weights=(2, 1))
# criterion = weighted_cross_entropy(weights=(10, 1))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)


# Train the network
epochs = 200
train_loss = []
val_loss = []

for epoch in range(epochs):

    n_batches = len(trainloader)
    print(f"Epoch {epoch+1}/{epochs}")

    # Progress bar
    pbar = tf.keras.utils.Progbar(target=n_batches)

    # Set the model to training mode
    model.train(True)

    # looping over batches
    for i, data in enumerate(trainloader, start=0):

        # get the inputs and labels for each batch
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        # loss = criterion(outputs, labels)  # For BCEWithLogitsLoss
        loss = criterion(outputs, torch.sum(labels, dim=1).long())
        # loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.update(i, values=[("loss", loss.item())])

    # Save the loss for this epoch
    train_loss.append(loss.item())

    # Set the model to evaluation mode
    model.train(False)
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(validationloader, start=0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # loss = criterion(outputs, labels)  # For BCEWithLogitsLoss
            loss = criterion(
                outputs, torch.sum(labels, dim=1).long()
            )  # For CrossEntropyLoss
            running_loss += loss.item()
        val_loss.append(running_loss / len(validationloader))

    # Upadte progress bar
    pbar.update(n_batches, values=[("val_loss", val_loss[-1])])

    del inputs, labels, outputs, loss

# Save the model
MODEL_NAME = "UNet-noctiluca-2023-02-07"

DIR = "/data/copepod-tracking/models/" + MODEL_NAME + "/"
os.makedirs("/data/copepod-tracking/models/" + MODEL_NAME, exist_ok=True)

torch.save(model.state_dict(), DIR + MODEL_NAME + ".pth")

# Save history
plt.figure()
plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.legend()
plt.savefig(DIR + MODEL_NAME + ".png")


# Plot example predictions
model.eval()
with torch.no_grad():
    for i, data in enumerate(validationloader, start=0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # outputs = torch.sigmoid(outputs)
        outputs = torch.softmax(outputs, dim=1)
        break

# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(inputs[0, 0, :, :].cpu().numpy(), cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(outputs[0, 0, :, :].cpu().numpy(), cmap="gray")

# plt.figure(figsize=(10, 10))
# plt.subplot(1, 3, 1)
# plt.imshow(inputs[0, 0, :, :].cpu().numpy(), cmap="gray")
# plt.subplot(1, 3, 2)
# plt.imshow(outputs[0, 0, :, :].cpu().numpy(), cmap="gray")
# plt.subplot(1, 3, 3)
# plt.imshow(outputs[0, 1, :, :].cpu().numpy(), cmap="gray")

plt.figure(figsize=(10, 10))
plt.subplot(1, 4, 1)
plt.imshow(inputs[0, 0, :, :].cpu().numpy(), cmap="gray")
plt.subplot(1, 4, 2)
plt.imshow(outputs[0, 0, :, :].cpu().numpy(), cmap="gray")
plt.subplot(1, 4, 3)
plt.imshow(outputs[0, 1, :, :].cpu().numpy(), cmap="gray")
plt.subplot(1, 4, 4)
plt.imshow(outputs[0, 2, :, :].cpu().numpy(), cmap="gray")

# Save figure
plt.savefig(DIR + MODEL_NAME + "-predictions.png")

# %%
