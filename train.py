import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Load embeddings and precompute tensors
embeddings_list = []
for file in os.listdir("embeddings"):
    array = np.load(f"embeddings/{file}")
    if array.shape[0] > 62:
        array = array[:62]  # Ensure shape[0] is 62
    embeddings_list.append(torch.tensor(array, dtype=torch.float32))


# Define the neural network
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


# Fitness function
def fitness_function(prev_output, curr_output):
    difference = torch.sum((curr_output - prev_output) ** 2)
    moderation = torch.sum(curr_output**2)

    # Encourage differences but not too drastic
    encouragement_factor = 0.5  # Adjust this factor to control the level of change
    moderation_factor = 0.01  # Adjust this factor to penalize drastic changes

    fitness = -difference + encouragement_factor * moderation_factor * moderation

    return fitness


# Reinforcement Learning Algorithm
def train_network(
    embeddings,
    epochs=8,
    learning_rate=0.001,
    batch_size=200,
    save_dir="model_checkpoints",
):
    device = torch.device("cpu")
    model = EmbeddingNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        total_fitness = 0
        model.train()

        # Randomly sample a subset of embeddings for this epoch
        sampled_embeddings = random.sample(embeddings, min(batch_size, len(embeddings)))

        for index, embedding in enumerate(sampled_embeddings):
            embedding = embedding.to(device)
            prev_output = torch.zeros((1, 16), dtype=torch.float32).to(device)

            for i in range(1, embedding.shape[0]):
                prev_frame = embedding[i - 1].unsqueeze(0)
                curr_frame = embedding[i].unsqueeze(0)

                optimizer.zero_grad()
                curr_output = model(prev_frame, curr_frame)
                fitness = fitness_function(prev_output, curr_output)
                loss = -fitness

                loss.backward()
                optimizer.step()

                prev_output = curr_output.detach()
                total_fitness += fitness.item()

        print(f"Epoch {epoch+1}/{epochs}, Total Fitness: {total_fitness}")
        # Save the model and optimizer state at each epoch
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_fitness": total_fitness,
            },
            checkpoint_path,
        )
        print(f"Model saved to {checkpoint_path}")

    return model


# Train the network
trained_model = train_network(embeddings_list)
