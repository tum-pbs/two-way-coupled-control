from typing import Iterable, Tuple
import torch
from TwoWayCouplingSimulation import *
from Probes import Probes
from collections import defaultdict
from phi.torch.flow import *
import os


def update_inputs(past_inputs: torch.Tensor, present_inputs: torch.Tensor, *control_effort: torch.Tensor) -> torch.tensor:
    """
    Create a new tensor containing inputs from present and past

    Params:
        past_inputs: network inputs from previous timesteps
        present_inputs: network inputs of current timestep
        *control_effort: outputs of network at current timestep

    Returns:
        new_past_inputs: updated past inputs

    """
    new_past_inputs = torch.cat((
        past_inputs[1:, :, :],
        torch.cat((present_inputs.view(-1), torch.cat(control_effort))).view(1, 1, -1)),
        dim=0)
    return new_past_inputs


def extract_inputs(sim: TwoWayCouplingSimulation, probes: Probes, x_objective: tuple, angle_objective: tuple, ref_vars: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract inputs that will be used on network model

    Params:
        sim: simulation object
        probes: probes manager
        x_objective : final destination in xy space
        angle_objective: angle that the box should go to
        ref_vars: reference variables for non dimensionalizing/normalizing inputs

    Returns:
        model_inputs: tensor containing inputs for model
        loss_inputs: tensor containing inputs for loss

    """
    if not ref_vars: ref_vars = defaultdict(lambda: 1)
    model_inputs = torch.cat(
        [
            sim.velocity.x.sample_at(probes.get_points_as_tensor()).native() / ref_vars['velocity'],
            sim.velocity.y.sample_at(probes.get_points_as_tensor()).native() / ref_vars['velocity'],
            sim.obstacle.velocity.native() / ref_vars['velocity'],
            (x_objective - sim.obstacle.geometry.center).native() / ref_vars['length'],
            sim.fluid_force.native() / ref_vars['force'],
            # (angle_objective - sim.obstacle.geometry.angle).native().view(1) / ref_vars['angle'],
            # math.sum(sim.fluid_torque).native().view(1) / ref_vars['torque'],
        ])
    loss_inputs = torch.cat(
        ((x_objective - sim.obstacle.geometry.center).native() / ref_vars['length'],
         sim.obstacle.velocity.native() / ref_vars['velocity'],
         #  (angle_objective - sim.obstacle.geometry.angle).native().view(1) / ref_vars['angle'],
         #  sim.obstacle.angular_velocity.native().view(1) / (ref_vars['angle'] / ref_vars['time'])
         ))
    return model_inputs.view(1, 1, -1), loss_inputs.view(1, -1)


def calculate_loss(loss_inputs: math.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate loss

    Params:
        loss_inputs: inputs needed for calculating loss

    Returns:
        loss: loss value
        spatial_term: loss term that accounts for space position of obstacle
        velocity_term: loss term that accounts for obstacle velocity
        ang_term: loss term that accounts for obstacle angle
        ang_vel_term: loss term that accounts for obstacle angular velocity

    """
    x_error = loss_inputs[:, :, 0:2]
    obs_velocity = loss_inputs[:, :, 2:4]
    # ang_error = loss_inputs[:, :, 4]
    # angular_velocity = loss_inputs[:, :, 5]
    # control_force = loss_inputs[:, :, 6:8]
    # control_torque = loss_inputs[:, :, 8]
    spatial_term = 15 * torch.sum(x_error**2)
    # Velocity term and angle term are pronounced only when spatial error is low
    velocity_term = 1 * torch.sum(obs_velocity**2 / (x_error**2 * 0.5 + 1))
    # ang_term = 10 * torch.sum(ang_error**2 / (torch.sum(x_error**2) * 0.5 + 1))
    # Angular velocity term is only pronounced when angular error is low
    # ang_vel_term = 2 * torch.sum(angular_velocity**2 / (ang_error**2 * 0.5 + 1))
    loss = spatial_term + velocity_term  # + ang_term + ang_vel_term
    # return loss, spatial_term, velocity_term, ang_term, ang_vel_term
    return loss, spatial_term, velocity_term, 0, 0


def prepare_export_folder(path: str, initial_step: int):
    """
    Prepare export folder by deleting export files from snapshots after the initial step.
    If folder does not exist, then it will be created.

    Params:
        path: path which will be analyzed
        initial_step: all steps after this value will be deleted

    """
    print(" Preparing export folder ")
    if not os.path.exists(path): os.mkdir(path)
    if not os.path.exists(path + '/data/'): os.mkdir(path + '/data/')
    if not os.path.exists(path + 'tensorboard/'): os.mkdir(path + 'tensorboard/')
    path += 'data/'
    folders = [folder for folder in os.listdir(path) if "." not in folder]
    for folder in folders:
        files = os.listdir(f"{path}/{folder}/")
        for file in files:
            if int(file.split("_")[-1][:4]) >= initial_step:
                os.remove(f"{path}/{folder}/{file}")
    print(" Export folder ready ")


def log_value(filepath: str, values: Iterable, should_delete: bool):
    """
    Log a value to a file. Useful for debug purposes.

    Params:
        filepath: path to file that will be used for logging
        values: values that will be logged
        should_delete: if True then the the file in filepath will be deleted before logging

    """
    if should_delete:
        try:
            os.remove(filepath)
        except:
            pass
    with open(filepath, 'a+') as f:
        for value in values:
            f.write(f"{value} ")
