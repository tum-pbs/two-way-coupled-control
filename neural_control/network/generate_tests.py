import os
import numpy as np
import json
from Dataset import Dataset
from InputsManager import InputsManager

if __name__ == "__main__":
    n_tests = 10
    # Delete previous test file
    test_filepath = "/home/ramos/work/PhiFlow2/PhiFlow/neural_control/tests.json"
    try:
        os.remove(test_filepath)
    except:
        print("Could not delete previous test file")
        pass
    # Generate destinations and angles
    randomGenerator = np.random.RandomState()
    randomGenerator.seed(999)
    export_dict = {}
    for test_i in range(n_tests):
        key = f"test{test_i}"
        export_dict[key] = {}
        values = randomGenerator.rand(3)
        x = values[0] * 40 + 40
        y = values[1] * 20 + 20
        angle = values[2] * 2 * 3.14159 - 3.14159
        # Store them in export dict
        export_dict[key]["positions"] = [[x, y]]
        export_dict[key]["angles"] = [angle]
        export_dict[key]["i"] = [0]
        export_dict[key]["n_steps"] = 1000

    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json")
    dataset = Dataset(inp.supervised_datapath, inp.tvt_ratio)
    dataset.set_mode("validation")
    export_dict = {}
    for case in range(dataset.n_cases):
        # inputs, labels = dataset.get_values_by_case(case)
        # Dump them into json file
        destination = np.array(dataset.get_destination(case)).tolist()
        export_dict[f"test{case}"] = dict(
            positions=[destination],
            i=[0],
            n_steps=2001,
            angles=(randomGenerator.rand(1) * 2 * 3.14159 - 3.14159).tolist()
        )
    with open(test_filepath, "a+") as f:
        json.dump(export_dict, f, indent="    ")
    print("Done")
