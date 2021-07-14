import torch
import torch.nn as nn
import torch.nn.functional as F
import os

torch.manual_seed(1)


class NeuralController(nn.Module):
    def __init__(self, n_features, id, n_outputs=2):
        super(NeuralController, self).__init__()
        self.id = id
        if id == 'lstm':
            h = 20
            self.present_layer = nn.Linear(n_features - n_outputs, h)
            self.past_layer = nn.LSTM(n_features, h, 1)
            self.output_layers = nn.ModuleList()
            self.output_layers += [nn.Linear(2 * h, n_outputs)]
        if id == 'fc0':
            self.layers = nn.ModuleList()
            n_out1 = 59
            n_out2 = 59
            self.layers += [nn.Linear(n_features, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
        if id == 'fc1':
            self.layers = nn.ModuleList()
            n_out1 = 47
            n_out2 = 47
            self.layers += [nn.Linear(n_features, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
        if id == 'fc2':
            self.layers = nn.ModuleList()
            n_out1 = 38
            n_out2 = 38
            self.layers += [nn.Linear(n_features, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
        if id == 'fc3':
            self.layers = nn.ModuleList()
            n_out1 = 31
            n_out2 = 31
            self.layers += [nn.Linear(n_features, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
            # for i in range(n_layers - 1):
            #     self.layers += [nn.Linear(int((n_features) / (2**i)), int((n_features) / (2**(i + 1))))]
            # self.layers += [nn.Linear(int((n_features) / (2**(i + 1))), n_outputs)]

    def forward(self, x_present, x_past=None) -> torch.Tensor:
        if self.id == 'lstm':
            l_past = self.past_layer(x_past)[0][-1:, :, :]  # Get last latent space of last cell
            l_present = self.present_layer(x_present)
            l_present = F.leaky_relu(l_present)
            output = torch.cat((l_past, l_present), -1)[0]
            for layer in self.output_layers[:-1]:
                output = F.leaky_relu(layer(output))
            output = self.output_layers[-1](output)
            return output
        if 'fc' in self.id:
            x = torch.cat((x_past.view(-1), x_present.view(-1))).view(1, -1) if x_past is not None else x_present.view(1, -1)
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            x = self.layers[-1](x)
            return x


if __name__ == '__main__':

    # Initialize model
    from InputsManager import InputsManager
    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json")
    inp.calculate_properties()
    # model = NeuralController(inp.n_past_features, 'lstm', 3)
    model = NeuralController(inp.n_past_features * inp.past_window + inp.n_present_features, f'fc{inp.past_window}', 2)

    # Print model's state_dict and total params
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total amount of trainable parameters: {total_params}")
    # Export
    torch.save(model, inp.nn_model_path + f"model_fc_pw{inp.past_window:02d}.pth")
