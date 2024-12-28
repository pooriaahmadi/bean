import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(
            128 * 4 * 4, 512
        )  # Adjust dimensions based on input size and conv layers
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(128, 16)

    def forward(self, prev_frame, curr_frame):
        # Reshape input frames to 32x32 and concatenate along the channel dimension
        prev_frame = prev_frame.view(-1, 1, 32, 32)
        curr_frame = curr_frame.view(-1, 1, 32, 32)
        x = torch.cat(
            (prev_frame, curr_frame), dim=1
        )  # Concatenate along channel dimension

        # Apply convolutional and pooling layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        output = self.fc3(x)
        return output


# Load the checkpoint
def load_checkpoint(checkpoint_path):
    model = EmbeddingNetwork()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    return model


# Prepare the embedding
def prepare_embedding(embedding_path):
    array = np.load(embedding_path)
    embedding = torch.tensor(array, dtype=torch.float32)
    return embedding


def test_embedding_and_save(model, embedding, output_path):
    device = torch.device("cpu")
    embedding = embedding.to(device)
    prev_output = torch.zeros((1, 16), dtype=torch.float32).to(device)

    outputs = []

    for i in range(1, embedding.shape[0]):
        prev_frame = embedding[i - 1].unsqueeze(0)
        curr_frame = embedding[i].unsqueeze(0)

        curr_output = model(prev_frame, curr_frame)
        outputs.append(curr_output.cpu().detach().numpy())
        prev_output = curr_output.detach()
        print(f"Frame {i}, Output: {curr_output}")

    # Save all outputs to a .npy file
    outputs_array = np.array(outputs)
    np.save(output_path, outputs_array)
    print(f"Outputs saved to {output_path}")


# Example usage
checkpoint_path = "model_checkpoints/model_epoch_2.pth"
embedding_path = "cityofstars.npy"
output_path = "outputs/cityofstars.npy"


# Load the model from the checkpoint
model = load_checkpoint(checkpoint_path)

# Prepare the embedding
embedding = prepare_embedding(embedding_path)

# Test the embedding
test_embedding_and_save(model, embedding, output_path)
