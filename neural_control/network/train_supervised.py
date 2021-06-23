from natsort import natsorted
import os
import logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
# import nn
# import tensorflow.python.keras as keras
import shutil
from model_torch import NeuralController
from inputsManager import inputsManager
from dataset import Dataset
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

    inp = inputsManager("/home/ramos/work/PhiFlow2/PhiFlow/myscripts/inputs.json")
    device = torch.device("cuda:0")
    # for past_window in range(4):
    shutil.rmtree(inp.nn_model_path + '/tensorboard/', ignore_errors=True)
    root = "/home/ramos/work/PhiFlow2/PhiFlow/storage/"
    models_file = natsorted(file for file in os.listdir(root) if (".pth" in file) and ("lstm" not in file))
    dataset = Dataset(inp.supervised_datapath, inp.tvt_ratio)
    for model_path in models_file:
        model = torch.load(f'{root}/{model_path}').to(device)
        torch.cuda.empty_cache()
        past_window = int(model_path.split('pw')[1][:2])
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n Total amount of trainable parameters: {total_params}")
        dataset.set_past_window_size(past_window)
        dataset.set_mode('training')
        dataloader = torch.utils.data.DataLoader(dataset, **inp.dataloader_params)
        learning_rate = inp.learning_rate_supervised
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        writer = tb.SummaryWriter(inp.nn_model_path + '/tensorboard/')
        # tbm = tensorboard_manager()
        # Loop over epochs
        for epoch in range(inp.n_epochs):
            if (epoch + 1) % 100 == 0:
                learning_rate /= 2
                print(f'Learning rate now {learning_rate}')
            # Training
            dataset.set_mode('training')
            training_loss = 0
            for batch_counter, (x_local, y_local) in enumerate(dataloader):
                batch_counter += 1
                x_local, y_local = x_local.to(device), y_local.to(device)
                y_predict = model(x_local)
                local_loss = calculate_loss(y_local, y_predict)
                local_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                training_loss += local_loss.detach().cpu().numpy()
            training_loss /= batch_counter
            # Validation
            dataset.set_mode('validation')
            with torch.set_grad_enabled(False):
                validation_loss = 0
                for batch_counter, (x_local, y_local) in enumerate(dataloader):
                    # Transfer to GPU
                    x_local, y_local = x_local.to(device), y_local.to(device)
                    y_predict = model(x_local)
                    local_loss = calculate_loss(y_local, y_predict)
                    validation_loss += local_loss.detach().cpu().numpy()
                validation_loss /= batch_counter
                # Log scalars
                writer.add_scalars('Loss', {f'pw{past_window:02d}/Training': training_loss,
                                            f'pw{past_window:02d}/Validation': validation_loss}, global_step=epoch)
                # Log weights
                weigths, biases = get_weights_and_biases(model)
                for var in (weigths, biases):
                    for tag, value in var.items():
                        writer.add_histogram(f'pw{past_window:02d}/{tag}', value, global_step=epoch)
                writer.flush()
            if epoch % inp.export_stride == 0:
                torch.save(model, f"{inp.nn_model_path}/model_pw{past_window:02d}_trained.pth")
            print(past_window, epoch)
        writer.close()
    print('Done')
