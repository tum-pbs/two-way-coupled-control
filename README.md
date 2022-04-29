# Control of Two-way Coupled Fluid Systems with Differentiable Solvers

This is a modified version of the differentiable solver  [Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow) designed to investigate how differentiable solvers can be used to train neural networks to act as controllers in an unsupervised way for a fluid system with two-way coupling.

<!-- TODO Add a brief theoretical explanation -->
# Installation
It is recommended to use a virtual environment such as [conda](https://docs.conda.io/en/latest/ ) in order to properly install the required packages.
```
conda create --name phiflow python=3.7
conda activate phiflow
```

The specific implementations for two-way coupling used the PyTorch framework, which can be installed with
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Navigate to the folder you would like to use to store the code and clone this repository.
```
git clone https://github.com/brenerrr/PhiFlow.git ./
```

Lastly install all packages.
```
cd PhiFlow; pip install . ./
```

Next verify that Φ<sub>Flow</sub> is correctly installed and can access PyTorch.
```
python tests/verify.py
```

Inputs required to run most scripts are grouped in the file *neural_control/inputs.json*.
# Generate Initial Conditions
Simulations used as initial conditions are performed by executing
```
python neural_control/mic/generate_ic.py
```
Inputs necessary for generating initial conditions for training and test environments can be found in the "simulation" section of *inputs.json*. All initial conditions used in our paper can be found on *storage/ics/*.
# Training
Training neural networks as controllers using the differentiable simulator is achieved by executing
```
python neural_control/neural_networks/train_unsupervised.py path/to/storage/folder/
```

Intermediate models are saved during training before all iterations are performed as *trained_model_####.pt*, where #### is a model index.

All parameters for training are set in *inputs.json*, especially in the "unsupervised" session.

# Tests
Running test simulations can be performed by

```
python neural_control/testing/test_networks.py path/to/model/folder/ model_index tests_id
```

The table below summerizes all tests available with their respective IDs.


| Tag                    | Test ID | DoF | Re   | N Simulations | Buoyancy | Inflow | Agitator |
| ---------------------- | ------- | --- | ---- | ------------- | -------- | ------ | -------- |
| Ba2                    | 1       | 2   | 1000 | 20            |          |        |          |
| Buo2_OneSimulation     | 2       | 2   | 1000 | 1             | ✔        |        |          |
| Ba3                    | 3       | 3   | 1000 | 20            |          |        |          |
| In3_OneSimulation      | 4       | 3   | 3000 | 1             |          | ✔      |          |
| InBuo3_OneSimulation   | 5       | 3   | 3000 | 1             | ✔        | ✔      |          |
| InBuoAg3_OneSimulation | 6       | 3   | 5000 | 1             | ✔        | ✔      | ✔        |
| In3                    | 7       | 3   | 3000 | 5             |          | ✔      |          |
| InBuo3                 | 8       | 3   | 3000 | 5             | ✔        | ✔      |          |
| InBuoAg3               | 9       | 3   | 5000 | 5             | ✔        | ✔      | ✔        |
| Buo2                   | 10      | 2   | 1000 | 5             | ✔        |        |          |

After running the desired tests, calculate the metrics with
```
python neural_control/misc/group_frames.py /path/to/model/folder/
python neural_control/misc/calculate_metrics.py /path/to/model/folder/
```

This will calculate all metrics and export them to /path/to/model/folder/tests/test#_#/metrics.json.

# Visualization
Simulations can be quickly visualized with an interactive data visualizer GUI by
```
python neural_control/visualization/data_visualizer.py
```

![Gui](gui.png)



When loading the GUI for the first time, after inserting the path to the directory containing the models folders, click on "Update paths" in order to update the models folder options menu.

The folder structure of the example above is set the following way
```
root path
│
└───diff_2dof
    │   inputs.json
    |   trained_model_0010.pt
    |
    └───tests
    |   │
    |   └───test2_10
    |       │   metrics.json
    |       |
    |       └───data
    |
    └───data (training data)
```


If "Test Data" is not ticked, the GUI will try to plot the "data" folder directly at the model folder (training data) instead of the one of the chosen test.
<!-- ## Fields
TODO Fields export used in the paper
``` bash
python neural_control/visualization/plot_fields.py 10 3900 --folders diff
python neural_control/visualization/plot_fields.py 7 5900 --folders diff
python neural_control/visualization/plot_fields.py 8 3900 --folders diff
python neural_control/visualization/plot_fields.py 9 5900 --folders diff
``` -->



