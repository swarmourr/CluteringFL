import torch
import torch.nn as nn
import pandas as pd

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GradientExtractor:
    def __init__(self, model_class, model_params, num_models):
        self.models = [model_class(**model_params) for _ in range(num_models)]
        self.model_names = [f'{i+1}' for i in range(num_models)]

    def extract_gradients(self):
        all_gradients = []
        for model, name in zip(self.models, self.model_names):
            dummy_input = torch.randn(1, model.fc1.in_features)
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()

            gradients_dict = {'model_name': name}
            for param_name, param in model.named_parameters():
                if param.grad is not None:
                    flattened_gradients = param.grad.detach().numpy().flatten()
                    for i, gradient in enumerate(flattened_gradients):
                        header = f"{param_name}_grad_{i}"
                        gradients_dict[header] = gradient

            all_gradients.append(gradients_dict)
            
            # Reset gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        df = pd.DataFrame(all_gradients)
        return df
