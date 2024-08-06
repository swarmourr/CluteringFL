import torch
import torch.nn as nn
import pandas as pd

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleModel()

# Function to extract and label weights
def extract_weights(model):
    weights_dict = {}
    for name, param in model.named_parameters():
        # Flatten the weights and store them in a dictionary
        flattened_weights = param.detach().numpy().flatten()
        for i, weight in enumerate(flattened_weights):
            # Create a descriptive header using the parameter name and index
            header = f"{name}_{i}"
            weights_dict[header] = weight

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(weights_dict, index=[0])
    return df

# Extract weights from the model
weights_df = extract_weights(model)

# Display the DataFrame
print(weights_df)
