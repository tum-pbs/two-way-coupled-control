from collections import defaultdict
import os
from turtle import position
from Dataset import Dataset
import numpy as np
import json
import matplotlib.pyplot as plt
from InputsManager import InputsManager

if __name__ == "__main__":
    randomGenerator = np.random.RandomState()
    randomGenerator.seed(900)
    tests = dict(
        test1=dict(
            initial_conditions_path="/home/ramos/phiflow/storage/baseline_simple_noinflow2/",
            n_simulations=20,
            positions=lambda: ((randomGenerator.rand(2) * 40 + 20).tolist(),),
            help_i=lambda: [-1],
            i=lambda: [-1],
            smoke=lambda: dict(
                on=False,
            ),
            n_steps=lambda: 1001,
            # angles=lambda: (randomGenerator.rand(1) * 2 * 3.14159 - 3.14159).tolist()
            angles=lambda: (0,)

        ),
        test2=dict(
            initial_conditions_path="/home/ramos/phiflow/storage/baseline_simple_noinflow_smalldt/",
            n_simulations=1,
            help_i=lambda: -1,
            export_stride=lambda: 50,
            n_steps=lambda: 8001,
            positions=lambda: [[30, 30, ], [30, 60], [50, 30], [50, 60]],
            i=lambda: [-1, 2000, 4000, 6000],
            angles=lambda: [np.pi / 4, -np.pi / 4, np.pi, 0],
            smoke=lambda: dict(
                on=True,
                xy=[[0.5, 0.1]],
                inflow=0.01,
                width=60,
                height=6,
                buoyancy=(0, 0.02),
            )
        ),
    )

    export_dict = {}
    for test_id, test_attrs in tests.items():
        export_dict[test_id] = {f"case{i}": dict(positions=test_attrs['positions']()) for i in range(test_attrs['n_simulations'])}
        # Add remaining attributes
        for case_key in export_dict[test_id]:
            for key, value in test_attrs.items():
                if key in ["initial_conditions_path", "n_simulations"]: continue  # add this to root of test
                export_dict[test_id][case_key][key] = value()
        export_dict[test_id]["initial_conditions_path"] = test_attrs["initial_conditions_path"]
        export_dict[test_id]["n_simulations"] = test_attrs["n_simulations"]

    # inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json", ["supervised", "nn_vars"])
    # dataset = Dataset(inp.supervised['dataset_path'], inp.supervised['tvt_ratio'], inp.nn_vars)
    # dataset.set_mode("validation")
    # dataset_destinations = [dataset.get_destination(i) for i in range(dataset.n_simulations)]
    # dataset_destinations = [[x.tolist(), y.tolist()] for x, y in dataset_destinations]
    # export_dict = defaultdict(dict)
    # tests = dict(
    #     test1=dict(
    #         initial_conditions_path="/home/ramos/phiflow/storage/baseline_simple_noinflow2/",
    #         positions=lambda: (randomGenerator.rand(2) * 40 + 20).tolist(),
    #         help_i=lambda: [-1],
    #         i=lambda: [-1],
    #         smoke=lambda: dict(
    #             on=False,
    #         ),
    #         n_steps=lambda: 1001,
    #         angles=lambda: (randomGenerator.rand(1) * 2 * 3.14159 - 3.14159).tolist()

    #     ),
    #     test2=dict(
    #         help_i=lambda: -1,
    #         initial_conditions_path="/home/ramos/phiflow/storage/baseline_simple_noinflow_smalldt/",
    #         export_stride=lambda: 50,
    #         n_steps=lambda: 8001,
    #         positions=lambda: [[30, 30, ], [30, 60], [50, 30], [50, 60]],
    #         i=lambda: [-1, 2000, 4000, 6000],
    #         angles=lambda: [np.pi / 4, -np.pi / 4, np.pi, 0],
    #         smoke=lambda: dict(
    #             on=True,
    #             xy=[[0.5, 0.1]],
    #             inflow=0.01,
    #             width=60,
    #             height=6,
    #             buoyancy=(0, 0.02),
    #         )
    #     ),
    # )
    # for test_id, test_attrs in tests.items():
    #     # Manual positions
    #     if "positions" in test_attrs:
    #         export_dict[test_id][f"case0"] = dict(positions=test_attrs["positions"]())
    #     # Dataset positions
    #     else:
    #         export_dict[test_id] = {f"case{case}": dict(positions=[destination]) for case, destination in enumerate(dataset_destinations)}
    #     # Add remaining attributes
    #     for case_key in export_dict[test_id]:
    #         for key, value in test_attrs.items():
    #             if key == "initial_conditions_path": continue  # add this to root of test_id
    #             export_dict[test_id][case_key][key] = value()
    #     export_dict[test_id]["initial_conditions_path"] = test_attrs["initial_conditions_path"]
    with open(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json", "w") as f:
        json.dump(export_dict, f, indent="    ")
    for test_id, test_attrs in export_dict.items():
        positions = [test_attrs[f"case{i}"]["positions"] for i in range(test_attrs["n_simulations"])]
        positions = np.array(positions)
        plt.plot(*positions.transpose(), '+', linestyle='None')
        plt.show()
    print("Done")
