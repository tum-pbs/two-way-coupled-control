from phi.math import PI
from natsort import natsorted
import numpy as np
import os
import logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
# import nn
# import tensorflow.python.keras as keras
import shutil
from NeuralController import NeuralController
from InputsManager import InputsManager
from Dataset import Dataset
import torch
import torch.utils.tensorboard as tb


def get_weights_and_biases(model: NeuralController):
    weights = {}
    biases = {}
    for i, layer in enumerate(model.layers):
        weights[f'layer_{i}_W'] = layer.weight.detach()
        biases[f'layer_{i}_b'] = layer.bias.detach()
    return weights, biases


def calculate_loss(y, y_predict):
    return torch.mean(torch.linalg.norm(y - y_predict, 2, -1))


if __name__ == '__main__':

    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json")
    inp.calculate_properties(True)
    device = torch.device("cuda:0")
    # for past_window in range(4):
    shutil.rmtree(inp.nn_model_path + '/tensorboard/', ignore_errors=True)
    root = "/home/ramos/work/PhiFlow2/PhiFlow/storage/"
    models_file = natsorted(file for file in os.listdir(root) if (".pth" in file) and ("lstm" not in file))
    models_file = ["model_lstm_only_translation.pth"] * 3
    ref_vars = dict(
        velocity=inp.inflow_velocity,
        length=inp.obs_width,
        force=inp.obs_mass * inp.max_acc,
        angle=PI,
        torque=inp.obs_inertia * inp.max_ang_acc,
        time=inp.obs_width / inp.inflow_velocity,
        ang_velocity=inp.inflow_velocity / inp.obs_width
    )
    dataset = Dataset(inp.supervised_datapath, inp.tvt_ratio, ref_vars)
    for past_window, model_path in enumerate(models_file):
        model = torch.load(f'{root}/{model_path}').to(device)
        past_window += 1
        inp.past_window = past_window
        inp.calculate_properties(True)
        torch.cuda.empty_cache()
        # past_window = int(model_path.split('pw')[1][:2])
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n Total amount of trainable parameters: {total_params}")
        dataset.set_past_window_size(past_window)
        dataset.set_mode('training')
        dataloader = torch.utils.data.DataLoader(dataset, **inp.dataloader_params)
        learning_rate = inp.learning_rate_supervised
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        writer = tb.SummaryWriter(inp.nn_model_path + '/tensorboard/')
        decay = np.exp(np.log(0.5) / inp.learning_rate_decay_half_life)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay)
        # Loop over epochs
        for epoch in range(inp.n_epochs):
            # Training
            dataset.set_mode('training')
            training_loss = 0
            for batch_counter, (x_present, x_past, y_local) in enumerate(dataloader):
                batch_counter += 1
                y_local = y_local.to(device)
                x_present, x_past = x_present.to(device), x_past.to(device)
                x_past = x_past.view(x_present.shape[0], past_window, inp.n_past_features)
                x_past = torch.swapaxes(x_past, 0, 1)
                y_predict = model(
                    x_present.view(1, x_present.shape[0], inp.n_present_features),
                    x_past)
                local_loss = calculate_loss(y_local, y_predict)
                local_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                training_loss += local_loss.detach().cpu().numpy()
            training_loss /= batch_counter
            lr_scheduler.step()
            # Validation
            if epoch % inp.export_stride == 0:
                dataset.set_mode('validation')
                with torch.set_grad_enabled(False):
                    validation_loss = 0
                    for batch_counter, (x_present, x_past, y_local) in enumerate(dataloader):
                        batch_counter += 1
                        y_local = y_local.to(device)
                        x_present, x_past = x_present.to(device), x_past.to(device)
                        x_past = x_past.view(x_present.shape[0], past_window, inp.n_past_features)
                        x_past = torch.swapaxes(x_past, 0, 1)
                        y_predict = model(
                            x_present.view(1, x_present.shape[0], inp.n_present_features),
                            x_past)
                        local_loss = calculate_loss(y_local, y_predict)
                        validation_loss += local_loss.detach().cpu().numpy()
                    validation_loss /= batch_counter
                    # Log scalars
                    writer.add_scalars('Loss', {f'pw{past_window:02d}/Training': training_loss,
                                                f'pw{past_window:02d}/Validation': validation_loss}, global_step=epoch)
                    # Log weights
                    # weigths, biases = get_weights_and_biases(model)
                    # for var in (weigths, biases):
                    #     for tag, value in var.items():
                    #         writer.add_histogram(f'pw{past_window:02d}/{tag}', value, global_step=epoch)
                    writer.flush()
                    torch.save(model, f"{inp.nn_model_path}/model_pw{past_window:02d}_trained.pth")
            print(past_window, epoch)
        writer.close()
    print('Done')
