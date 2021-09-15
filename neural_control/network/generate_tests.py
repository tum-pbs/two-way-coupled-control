from collections import defaultdict
import os
import numpy as np
import json
from Dataset import Dataset
from InputsManager import InputsManager

if __name__ == "__main__":
    n_tests = 10
    # Delete previous test file
    # test_filepath = "/home/ramos/work/PhiFlow2/PhiFlow/neural_control/tests.json"
    # try:
    #     os.remove(test_filepath)
    # except:
    #     print("Could not delete previous test file")
    #     pass
    # Generate destinations and angles
    randomGenerator = np.random.RandomState()
    randomGenerator.seed(999)

    # export_dict = {}
    # for test_i in range(n_tests):
    #     key = f"test{test_i}"
    #     export_dict[key] = {}
    #     values = randomGenerator.rand(3)
    #     x = values[0] * 40 + 40
    #     y = values[1] * 20 + 20
    #     angle = values[2] * 2 * 3.14159 - 3.14159
    #     # Store them in export dict
    #     export_dict[key]["positions"] = [[x, y]]
    #     export_dict[key]["angles"] = [angle]
    #     export_dict[key]["i"] = [0]
    #     export_dict[key]["n_steps"] = 1000

    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json", ["supervised"])
    dataset = Dataset(inp.supervised['dataset_path'], inp.supervised['tvt_ratio'])
    dataset.set_mode("validation")
    export_dict = defaultdict(dict)
    export_dict["dataset_path"] = inp.supervised['dataset_path']
    export_dict["tvt_ratio"] = inp.supervised['tvt_ratio']
    tests = dict(
        test1=dict(
            help_i=lambda: 0,
            initial_conditions_path="/home/ramos/phiflow/storage/baseline_175x110_re3000/"
        ),
        test2=dict(
            help_i=lambda: 0,
            initial_conditions_path="/home/ramos/phiflow/storage/baseline_175x110_two_obstacles_re3000/"
        ),
        test3=dict(
            help_i=lambda: 0,
            initial_conditions_path="/home/ramos/phiflow/storage/baseline_175x110_two_obstacles_re8000/")
    )
    # Export initial conditions
    for label, test_attrs in tests.items():
        export_dict[label]["initial_conditions_path"] = test_attrs["initial_conditions_path"]
    # Loop through validation cases and export data necessary for test simulations
    for case in range(dataset.n_cases):
        destination = np.array(dataset.get_destination(case)).tolist()
        angles = (randomGenerator.rand(1) * 2 * 3.14159 - 3.14159).tolist()
        for label, test_attrs in tests.items():
            export_dict[label][f"test{case}"] = dict(
                positions=[destination],
                i=[0],
                n_steps=2001,
                help_i=test_attrs["help_i"](),
                angles=angles,
            )
    with open(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json", "w") as f:
        json.dump(export_dict, f, indent="    ")
    print("Done")
