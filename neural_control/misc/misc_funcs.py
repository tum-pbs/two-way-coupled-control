from typing import Iterable, Tuple
import torch
from neural_control.misc.TwoWayCouplingSimulation import *
from collections import OrderedDict, defaultdict
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


def extract_inputs(
    vars: list,
    sim: TwoWayCouplingSimulation,
    x_objective: tuple,
    angle_objective: tuple,
    ref_vars: dict = None,
    translation_only: bool = False,
    clamp: dict = {},
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract inputs that will be used on network model

    Params:
        vars: list of variables that will be used for extracting inputs
        sim: simulation object
        x_objective : final destination in xy space
        angle_objective: angle that the box should go to
        ref_vars: reference variables for non dimensionalizing/normalizing inputs
        translation_only: if true then inputs that are related with rotation will not be gathered
        clamp: dictionary of clamp values for each variable

    Returns:
        model_inputs: tensor containing inputs for model
        loss_inputs: tensor containing inputs for loss

    """

    if not ref_vars: ref_vars = defaultdict(lambda: 1)
    getter = dict(
        obs_vx=lambda: sim.obstacle.velocity.native()[0].view(1),
        obs_vy=lambda: sim.obstacle.velocity.native()[1].view(1),
        error_x=lambda: (x_objective[0] - sim.obstacle.geometry.center[0]).native().view(1),
        error_y=lambda: (x_objective[1] - sim.obstacle.geometry.center[1]).native().view(1),
        fluid_force_x=lambda: sim.fluid_force.native()[0].view(1),
        fluid_force_y=lambda: sim.fluid_force.native()[1].view(1),
        error_ang=lambda: (angle_objective - (sim.obstacle.geometry.angle - PI / 2)).native().view(1),
        fluid_torque=lambda: math.sum(sim.fluid_torque).native().view(1),
        obs_ang_vel=lambda: sim.obstacle.angular_velocity.native().view(1),
        control_force_x=lambda: None,
        control_force_y=lambda: None,
        control_torque=lambda: None,
    )
    ref_vars_hash = dict(
        obs_vx="velocity",
        obs_vy="velocity",
        error_x="length",
        error_y="length",
        fluid_force_x="force",
        fluid_force_y="force",
        control_force_x="force",
        control_force_y="force",
        error_ang="angle",
        fluid_torque="torque",
        control_torque="torque",
        obs_ang_vel="ang_velocity",
    )
    inputs = OrderedDict()
    for var in vars:
        value = getter[var]()
        if clamp.get(var, None):
            value = torch.clamp(value, -clamp[var], clamp[var])
        if value is not None: inputs[var] = value / ref_vars[ref_vars_hash[var]]
    # Rotate vector variables
    negative_angle = (sim.obstacle.geometry.angle - math.PI / 2.0).native()
    for key in inputs:
        if 'x' in key: xy = [inputs[key], inputs[key.replace('x', 'y')]]
        else: continue
        xy = rotate(torch.stack(xy), negative_angle)
        inputs[key], inputs[key.replace('x', 'y')] = xy
    # Transfer values of inputs to tensor
    model_inputs = torch.cat(list(inputs.values())).view(1, 1, -1)
    # Loss inputs
    loss_inputs = dict(
        error_x=inputs["error_x"],
        error_y=inputs["error_y"],
        obs_vx=inputs["obs_vx"],
        obs_vy=inputs["obs_vy"]
    )
    if not translation_only:
        loss_inputs['error_ang'] = inputs["error_ang"]
        # inputs["fluid_torque"],
        loss_inputs['obs_ang_vel'] = inputs["obs_ang_vel"]
    # loss_inputs =
    return model_inputs, loss_inputs


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
    error_xy = torch.cat((loss_inputs['error_x'], loss_inputs['error_y']), dim=-1)
    obs_velocity = torch.cat((loss_inputs['obs_vx'], loss_inputs['obs_vy']), dim=-1)
    delta_force = torch.cat((loss_inputs['d_control_force_x'], loss_inputs['d_control_force_y']), dim=-1)
    force = torch.cat((loss_inputs['control_force_x'], loss_inputs['control_force_y']), dim=-1)
    spatial_term = hyperparams['spatial'] * torch.sum(error_xy**2)
    dforce_term = hyperparams['delta_force'] * torch.sum(delta_force**2)
    force_term = hyperparams['force'] * torch.sum(force**2)
    # Other terms are pronounced only when spatial or angular error are low
    velocity_term = hyperparams['velocity'] * torch.sum(obs_velocity**2 / (torch.sum(error_xy**2, 2, keepdim=True)**2 * hyperparams['proximity'] + 1))
    if not translation_only:
        error_ang = loss_inputs['error_ang']
        angular_velocity = loss_inputs['obs_ang_vel']
        delta_torque = loss_inputs['d_control_torque']
        torque = loss_inputs['control_torque']
        ang_term = hyperparams['angle'] * torch.sum(error_ang**2)
        ang_vel_term = hyperparams['ang_velocity'] * torch.sum(angular_velocity**2 / (error_ang**2 * hyperparams['proximity'] + 1))
        torque_term = hyperparams['torque'] * torch.sum(torque**2)
        dtorque_term = hyperparams['delta_torque'] * torch.sum(delta_torque**2)
    else:
        ang_term = ang_vel_term = dtorque_term = torque_term = torch.tensor(0)

    loss = (
        spatial_term +
        velocity_term +
        ang_term +
        ang_vel_term +
        dtorque_term +
        dforce_term +
        force_term +
        torque_term
    ) / error_xy.shape[0]  # Take mean over rollouts

    loss_terms = dict(
        spatial=spatial_term,
        velocity=velocity_term,
        ang=ang_term,
        ang_vel=ang_vel_term,
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


def get_weights_and_biases(model):
    weights = {}
    biases = {}
    for i, layer in enumerate(model.layers):
        weights[f'layer_{i}_W'] = layer.weight.detach()
        biases[f'layer_{i}_b'] = layer.bias.detach()
    return weights, biases


def calculate_additional_forces(attrs: dict, n: int):
    """
    Calculate additional forces for simulation.

    Parameters:
        attrs: dictionary with attributes
    """
    additional_forces = torch.tensor((0.0, 0.0))
    force_function = dict(
        constant=lambda direction, t: torch.tensor(direction),
        rotating=lambda direction, t: torch.tensor((np.cos(t / 2000 * np.pi * 2), np.sin(t / 2000 * np.pi * 2))),
    )
    if attrs != {}:
        for force_parameters in attrs:
            if n <= force_parameters['start']: continue
            if n > force_parameters['start'] + force_parameters['duration']: continue
            n_local = n - force_parameters['start']
            additional_forces = additional_forces + force_function[force_parameters['mode']](force_parameters['direction'], n_local) * force_parameters['strength']
    return additional_forces
