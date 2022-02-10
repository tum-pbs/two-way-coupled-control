from phi.math import PI
from natsort import natsorted
import numpy as np
import os
import logging
from time import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
# import nn
# import tensorflow.python.keras as keras
import shutil
from NeuralController import NeuralController
import argparse
from InputsManager import InputsManager
from Dataset import Dataset
import torch
import torch.utils.tensorboard as tb
from misc_funcs import get_weights_and_biases


def calculate_loss(y, y_predict):
    # Return l2 loss
    return torch.sum((y - y_predict)**2) / y.shape[0]
    # return torch.mean(torch.linalg.norm(y - y_predict, 2, -1))


if __name__ == '__main__':
    seed = 10
    parser = argparse.ArgumentParser(description='Train nn in an online setting')
    parser.add_argument("export_path", help="data will be saved in this path")
    args = parser.parse_args()
    export_path = args.export_path + '/'
    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json", exclude=["online", "simulation"])
    inp.add_values(inp.supervised['dataset_path'] + "/inputs.json", only=['simulation'])
    inp.calculate_properties()
    if inp.device == "GPU":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    shutil.rmtree(export_path + '/tensorboard/', ignore_errors=True)

    label_vars = [var for var in inp.nn_vars if 'control' in var]
    dataset = Dataset(inp.supervised['dataset_path'], inp.supervised['tvt_ratio'], inp.nn_vars, label_vars, local=inp.supervised['local_coordinates'])
    local_str = '_local' if inp.supervised['local_coordinates'] else ''
    dataset.ref_vars["velocity"] = 2 * (dataset.stdd['obs_vx' + local_str] + dataset.stdd['obs_vy' + local_str]) / 2
    dataset.ref_vars["length"] = 2 * (dataset.stdd['error_x' + local_str] + dataset.stdd['error_y' + local_str]) / 2
    dataset.ref_vars["force"] = 2 * (dataset.stdd['control_force_x' + local_str] + dataset.stdd['control_force_y' + local_str]) / 2
    if not inp.translation_only:
        dataset.ref_vars["torque"] = 2 * dataset.stdd['control_torque']
        dataset.ref_vars["angle"] = 2 * dataset.stdd['error_ang']
        dataset.ref_vars["ang_velocity"] = 2 * dataset.stdd['obs_ang_vel']
    else:
        dataset.ref_vars["torque"] = 0
        dataset.ref_vars["angle"] = 0
        dataset.ref_vars["ang_velocity"] = 0
    torch.manual_seed(seed)
    model = NeuralController(
        f"{inp.architecture}{inp.past_window}",
        2 if inp.translation_only else 3,
        inp.n_present_features,
        inp.n_past_features,
        inp.past_window).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total amount of trainable parameters: {total_params}")
    dataset.set_past_window_size(inp.past_window)
    dataloader = torch.utils.data.DataLoader(dataset, **inp.supervised['dataloader_params'])
    learning_rate = inp.supervised['learning_rate']
    optimizer_func = getattr(torch.optim, inp.supervised['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=inp.supervised['learning_rate'])
    writer = tb.SummaryWriter(export_path + '/tensorboard/')
    decay = np.exp(np.log(0.5) / inp.supervised['lr_half_life'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay)
    inp.ref_vars = dataset.ref_vars
    inp.training_dt = inp.simulation['dt']
    inp.bypass_tanh = True
    inp.export(export_path + "/inputs.json")
    i = 0
    epoch = 0
    last_time = time()
    export_counter = 1
    previous_i = 0
    while i < inp.supervised['n_iterations']:
        # Training
        dataset.set_mode('training')
        training_loss = 0
        n_batches = 0
        for i_minibatch, (x_present, x_past, y_local) in enumerate(dataloader):
            y_local = y_local.to(device)
            x_present, x_past = x_present.to(device), x_past.to(device)
            # x_past = x_past.view(x_present.shape[0], inp.past_window, inp.n_past_features)
            # x_past = torch.swapaxes(x_past, 0, 1)
            y_predict = model(
                # x_present.view(1, x_present.shape[0], inp.n_present_features),
                x_present,
                x_past,
                inp.bypass_tanh)
            local_loss = calculate_loss(y_local, y_predict)
            local_loss.backward()
            optimizer.step()
            i += 1
            if i % inp.supervised['model_export_stride'] == 0:
                torch.save(model, f"{export_path}/trained_model_{export_counter:04d}.pth")
                export_counter += 1
            optimizer.zero_grad()
            lr_scheduler.step()
            training_loss += local_loss.detach() * y_predict.shape[0]  # Eliminate mean computation for now
            n_batches += y_predict.shape[0]
            # print(i_minibatch)
        epoch += 1
        training_loss /= n_batches
        # Calculate time left
        current_time = time()
        steps_performed = i - previous_i
        previous_i = i
        steps_left = inp.supervised['n_iterations'] - i
        speed = steps_performed / (current_time - last_time)
        time_left = steps_left / speed / 3600
        time_left_hours = int(time_left)
        time_left_minutes = int((time_left - time_left_hours) * 60)
        print(f"Time left: {time_left_hours:d}h {time_left_minutes:d} min. Current backprop i: {i}")
        last_time = current_time
        steps_performed = i
        # Validation
        dataset.set_mode('validation')
        with torch.no_grad():
            validation_loss = 0
            n_batches = 0
            for i_minibatch, (x_present, x_past, y_local) in enumerate(dataloader):
                y_local = y_local.to(device)
                x_present, x_past = x_present.to(device), x_past.to(device)
                # x_past = x_past.view(x_present.shape[0], inp.past_window, inp.n_past_features)
                # x_past = torch.swapaxes(x_past, 0, 1)
                y_predict = model(
                    # x_present.view(1, x_present.shape[0], inp.n_present_features),
                    x_present,
                    x_past,
                    inp.bypass_tanh)
                local_loss = calculate_loss(y_local, y_predict)
                validation_loss += local_loss * y_predict.shape[0]
                n_batches += y_predict.shape[0]
            validation_loss /= n_batches
            # Log scalars
            writer.add_scalars(
                'Loss',
                {f'Training': training_loss,
                 f'Validation': validation_loss}, global_step=epoch)
            # Log weights
            weigths, biases = get_weights_and_biases(model)
            for var in (weigths, biases):
                for tag, value in var.items():
                    writer.add_histogram(f'{tag}', value, global_step=epoch)
            writer.flush()
    # print(inp.past_window, epoch)
    writer.close()
    print('Done')
