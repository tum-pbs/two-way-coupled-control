from collections import defaultdict
import os
from Dataset import Dataset
import numpy as np
import json
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

    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json", ["supervised", "nn_vars"])
    dataset = Dataset(inp.supervised['dataset_path'], inp.supervised['tvt_ratio'], inp.nn_vars)
    dataset.set_mode("validation")
    export_dict = defaultdict(dict)
    export_dict["dataset_path"] = inp.supervised['dataset_path']
    export_dict["tvt_ratio"] = inp.supervised['tvt_ratio']
    tests = dict(
        test1=dict(
            help_i=lambda: -1,
            initial_conditions_path=lambda: "/home/ramos/phiflow/storage/baseline_simple_noinflow2/",
            smoke=dict(
                on=lambda: False,
            ),
            n_steps=lambda: 1001,
        ),
        test2=dict(
            help_i=lambda: -1,
            initial_conditions_path=lambda: "/home/ramos/phiflow/storage/baseline_simple_noinflow_smalldt/",
            n_steps=lambda: 2001,
            smoke=dict(
                on=lambda: True,
                xy=lambda: [[0.3, 0.1], [0.5, 0.1], [0.7, 0.1]],
                inflow=lambda: 0.01,
                radius=lambda: 5,
                buoyancy=lambda: (0, 0.01),
            )
        ),
        # test3=dict(
        #     help_i=lambda: 0,
        #     initial_conditions_path="/home/ramos/phiflow/storage/baseline_175x110_two_obstacles_re8000/")
    )
    # Export initial conditions
    for label, test_attrs in tests.items():
        export_dict[label]["initial_conditions_path"] = test_attrs["initial_conditions_path"]()
    # Loop through validation cases and export data necessary for test simulations
    for label, test_attrs in tests.items():
        for case in range(dataset.n_cases):
            destination = np.array(dataset.get_destination(case)).tolist()
            angles = (randomGenerator.rand(1) * 2 * 3.14159 - 3.14159).tolist()
            export_dict[label][f"test{case}"] = dict(
                positions=[destination],
                i=[-1],
                n_steps=[test_attrs["n_steps"]()],
                angles=angles,
            )
            for key, value in test_attrs.items():
                if key == "smoke":
                    export_dict[label][f"test{case}"]["smoke"] = {}
                    for smoke_key, smoke_value in value.items():
                        export_dict[label][f"test{case}"][key][smoke_key] = smoke_value()
                else:
                    export_dict[label][f"test{case}"][key] = value()
    with open(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json", "w") as f:
        json.dump(export_dict, f, indent="    ")
    print("Done")
