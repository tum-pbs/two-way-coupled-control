import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class NeuralController(nn.Module):
    def __init__(self, id, n_outputs, n_present_features, n_past_features, past_window):
        torch.manual_seed(1)
        super(NeuralController, self).__init__()
        self.id = id
        self.past_window = past_window
        if 'lstm' in id:
            h = 20
            assert (n_past_features > 0)
            self.present_layer = nn.Linear(n_present_features, h)
            self.past_layer = nn.LSTM(n_past_features, h, 1)
            self.output_layers = nn.ModuleList()
            self.output_layers += [nn.Linear(2 * h, n_outputs)]
        if id == 'fc_only0':
            self.layers = nn.ModuleList()
            n_out1 = 59
            n_out2 = 59
            self.layers += [nn.Linear(n_present_features + n_past_features * past_window, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
        if id == 'fc_only1':
            self.layers = nn.ModuleList()
            n_out1 = 47
            n_out2 = 47
            self.layers += [nn.Linear(n_present_features + n_past_features * past_window, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
        if id == 'fc_only2':
            self.layers = nn.ModuleList()
            n_out1 = 38
            n_out2 = 38
            self.layers += [nn.Linear(n_present_features + n_past_features * past_window, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
        if id == 'fc_only3':
            self.layers = nn.ModuleList()
            n_out1 = 32
            n_out2 = 32
            self.layers += [nn.Linear(n_present_features + n_past_features * past_window, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
        if id == 'fc_only4':
            self.layers = nn.ModuleList()
            n_out1 = 26
            n_out2 = n_out1
            self.layers += [nn.Linear(n_present_features + n_past_features * past_window, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]
        if id == 'fc_only5':
            self.layers = nn.ModuleList()
            n_out1 = 22
            n_out2 = n_out1
            self.layers += [nn.Linear(n_present_features + n_past_features * past_window, n_out1)]
            self.layers += [nn.Linear(n_out1, n_out2)]
            self.layers += [nn.Linear(n_out2, n_outputs)]

    def forward(self, x_present, x_past=None, bypass_tanh=False) -> torch.Tensor:
        if 'fc' in self.id:
            x = torch.cat((x_past, x_present), dim=1).to(x_present.device) if x_past is not None else x_present.view(1, -1)
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            x = self.layers[-1](x)
            if not bypass_tanh:
                x = torch.tanh(x)
            return x


if __name__ == '__main__':

    # Initialize model
    from InputsManager import InputsManager
    inp = InputsManager("/home/ramos/phiflow/neural_control/inputs.json")
    inp.calculate_properties()
    model = NeuralController(
        f'{inp.architecture}{inp.past_window}',
        2 if inp.translation_only else 3,
        inp.n_present_features,
        inp.n_past_features,
        inp.past_window
    )
    # Print model's state_dict and total params
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total amount of trainable parameters: {total_params}")
    # # Export
    # torch.save(model, inp.nn_model_path + f"dummy{inp.past_window:02d}.pth")
