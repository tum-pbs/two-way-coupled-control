import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class NeuralController(nn.Module):
    def __init__(self, n_features, id, n_outputs=2):
        super(NeuralController, self).__init__()
        # self.layers = nn.ModuleList()
        # print('Initializing model')
        # n_out1 = 38
        # n_out2 = 38
        # self.layers += [nn.Linear(n_features, n_out1)]
        # self.layers += [nn.Linear(n_out1, n_out2)]
        # self.layers += [nn.Linear(n_out2, n_outputs)]

        # for i in range(n_layers-1):
        #     self.layers += [nn.Linear(int((n_features)/(2**i)), int((n_features)/(2**(i+1))))]
        # self.layers += [nn.Linear(int((n_features)/(2**(i+1))), n_outputs)]
        self.id = id
        if id == 'lstm':
            h = 20
            self.present_layer = nn.Linear(n_features - n_outputs, h)
            self.past_layer = nn.LSTM(n_features, h, 1)
            self.output_layers = nn.ModuleList()
            self.output_layers += [nn.Linear(2 * h, n_outputs)]

    def forward(self, x_past, x_present) -> torch.Tensor:
        if self.id == 'lstm':
            l_past = self.past_layer(x_past)[0][-1:, :, :]  # Get last latent space of last cell
            l_present = self.present_layer(x_present)
            l_present = F.leaky_relu(l_present)
            output = torch.cat((l_past, l_present), -1)[0]
            for layer in self.output_layers[:-1]:
                output = F.leaky_relu(layer(output))
            output = self.output_layers[-1](output)
            return output

    # def forward(self, x) -> torch.Tensor:
    #     x_input = x
    #     for layer in self.layers[:-1]:
    #         x_input = F.relu(layer(x_input))
    #     x_input = self.layers[-1](x_input)
    #     return x_input


if __name__ == '__main__':

    # Initialize model
    from InputsManager import InputsManager
    inp = InputsManager("/home/ramos/felix/PhiFlow/neural_obstacle_control/inputs.json")
    inp.calculate_properties()
    model = NeuralController(inp.n_past_features, 'lstm', 2)

    # Print model's state_dict and total params
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total amount of trainable parameters: {total_params}")
    # Export
    torch.save(model, inp.nn_model_path + "model.pth")
