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
        torch.cat((present_inputs.view(-1), torch.cat(control_effort).view(-1))).view(1, 1, -1)),
        dim=0)
    return new_past_inputs


def extract_inputs(sim: TwoWayCouplingSimulation, probes: Probes, x_objective: tuple, angle_objective: tuple, ref_vars: dict = None, translation_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract inputs that will be used on network model

    Params:
        sim: simulation object
        probes: probes manager
        x_objective : final destination in xy space
        angle_objective: angle that the box should go to
        ref_vars: reference variables for non dimensionalizing/normalizing inputs
        translation_only: if true then inputs that are related with rotation will not be gathered

    Returns:
        model_inputs: tensor containing inputs for model
        loss_inputs: tensor containing inputs for loss

    """

    if not ref_vars: ref_vars = defaultdict(lambda: 1)
    # Gather inputs
    probes_velocity = torch.stack([
        sim.velocity.x.sample_at(probes.get_points_as_tensor()).native(),
        sim.velocity.y.sample_at(probes.get_points_as_tensor()).native()
    ])
    error_xy = (x_objective - sim.obstacle.geometry.center).native()
    fluid_force = sim.fluid_force.native()
    obs_velocity = sim.obstacle.velocity.native()
    error_angle = (angle_objective - (sim.obstacle.geometry.angle - PI / 2)).native().view(1)
    fluid_torque = math.sum(sim.fluid_torque).native().view(1)
    ang_velocity = sim.obstacle.angular_velocity.native().view(1)
    # Transfer values to box local reference frame
    negative_angle = (sim.obstacle.geometry.angle - math.PI / 2.0).native()
    # negative_angle = torch.tensor(0)
    probes_velocity = rotate(probes_velocity, negative_angle)
    error_xy = rotate(error_xy, negative_angle)
    fluid_force = rotate(fluid_force, negative_angle)
    obs_velocity = rotate(obs_velocity, negative_angle)

    model_inputs = [
        probes_velocity[0] / ref_vars['velocity'],
        probes_velocity[1] / ref_vars['velocity'],
        obs_velocity / ref_vars['velocity'],
        error_xy / ref_vars['length'],
        fluid_force / ref_vars['force'],
    ]
    loss_inputs = [
        error_xy / ref_vars['length'],
        obs_velocity / ref_vars['velocity'],
    ]
    if not translation_only:
        model_inputs += [
            error_angle / ref_vars['angle'],
            fluid_torque / ref_vars['torque'],
            ang_velocity / ref_vars['ang_velocity']
        ]
        loss_inputs += [
            error_angle / ref_vars['angle'],
            ang_velocity / ref_vars['ang_velocity']
        ]
    return torch.cat(model_inputs).view(1, 1, -1), torch.cat(loss_inputs).view(1, -1)


def rotate(xy: torch.Tensor, angle: float):
    """
    Rotate coordinates xy by angle

    Params:
        xy: list of coordinates with dimension [2,n].
        angle: angle (radians) by which xy will be rotated

    Returns:
        rotated_cooridnates: xy rotated by angle with dimension [2,n]

    """
    device = torch.device("cuda:0") if xy.is_cuda else torch.device("cpu")
    matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]).to(device)
    rotated_xy = torch.matmul(matrix, xy)
    return rotated_xy


def calculate_loss(loss_inputs: math.Tensor, hyperparams: dict, translation_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate loss

    Params:
        loss_inputs: inputs needed for calculating loss
        hyperparams: hyperparameters of loss terms. Should have the following keys
            - spatial
            - velocity
            - angle
            - ang_velocity
            - delta_force
            - delta_torque
            - proximity
        translation_only: if True then additional terms won't be calculated

    Returns:
        loss: loss value
        spatial_term: loss term that accounts for space position of obstacle
        velocity_term: loss term that accounts for obstacle velocity
        ang_term: loss term that accounts for obstacle angle
        ang_vel_term: loss term that accounts for obstacle angular velocity

    """
    x_error = loss_inputs[:, :, 0:2]
    obs_velocity = loss_inputs[:, :, 2:4]
    delta_force = loss_inputs[:, :, 4:6]  # TODO CHECK THIS
    spatial_term = hyperparams['spatial'] * torch.sum(x_error**2)
    # Other terms are pronounced only when spatial or angular error is low
    velocity_term = hyperparams['velocity'] * torch.sum(obs_velocity**2 / (x_error**2 * hyperparams['proximity'] + 1))
    if not translation_only:
        ang_error = loss_inputs[:, :, 4:5]
        angular_velocity = loss_inputs[:, :, 5:6]
        delta_force = loss_inputs[:, :, 6:8]
        delta_torque = loss_inputs[:, :, 8:9]
        ang_term = hyperparams['angle'] * torch.sum(ang_error**2 / (torch.sum(x_error**2, 2, keepdim=True) * hyperparams['proximity'] + 1))
        ang_vel_term = hyperparams['ang_velocity'] * torch.sum(angular_velocity**2 / (ang_error**2 * hyperparams['proximity'] + 1))
        # Avoid abrupt changes
        torque_term = hyperparams['delta_torque'] * torch.sum(delta_torque**2)
        # force_term = 0.025 * torch.sum(delta_force**2)
        # torque_term = 0.025 * torch.sum(delta_torque**2)
    else:
        ang_term = ang_vel_term = torque_term = torch.tensor(0)
    force_term = hyperparams['delta_force'] * torch.sum(delta_force**2)
    loss = spatial_term + velocity_term + ang_term + ang_vel_term + force_term + torque_term
    loss_terms = dict(
        spatial=spatial_term,
        velocity=velocity_term,
        ang=ang_term,
        ang_vel=ang_vel_term,
        force=force_term,
        torque=torque_term
    )
    return loss, loss_terms


def prepare_export_folder(path: str, initial_step: int):
    """
    Prepare export folder by deleting export files from snapshots after the initial step.
    If folder does not exist, then it will be created.

    Params:
        path: path which will be analyzed
        initial_step: all steps after this value will be deleted

    """
    print(" Preparing export folder ")
    os.makedirs(os.path.abspath(path), exist_ok=True)
    os.makedirs(os.path.abspath(path) + '/data/', exist_ok=True)
    path += 'data/'
    folders = [folder for folder in os.listdir(os.path.abspath(path)) if "." not in folder]
    for folder in folders:
        files = os.listdir(os.path.abspath(f"{path}/{folder}/"))
        for file in files:
            if int(file.split("_")[-1][:4]) >= initial_step:
                os.remove(os.path.abspath(f"{path}/{folder}/{file}"))
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
