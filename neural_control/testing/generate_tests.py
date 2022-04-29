from collections import defaultdict
import os
from turtle import position
from Dataset import Dataset
import numpy as np
import json
from InputsManager import InputsManager

if __name__ == "__main__":
    root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../")
    randomGenerator = np.random.RandomState()
    tests = dict(
        test1=dict(
            seed=900,
            initial_conditions_path=root + "/storage/ics/training_2dof/",
            tag='Ba2',
            n_simulations=20,
            positions=lambda i: ((randomGenerator.rand(2) * 40 + 20).tolist(),),
            help_i=lambda *args: [-1],
            i=lambda *args: [-1],
            smoke=lambda *args: dict(
                on=False,
            ),
            n_steps=lambda *args: 1001,
            # angles=lambda *args: (randomGenerator.rand(1) * 2 * 3.14159 - 3.14159).tolist()
            angles=lambda *args: (0,)

        ),
        test2=dict(
            initial_conditions_path=root + "/storage/ics/Buo2/",
            n_simulations=1,
            tag='Buo2_OneSimulation',
            help_i=lambda *args: -1,
            export_stride=lambda *args: 20,
            n_steps=lambda *args: 8001,
            positions=lambda i: [[30, 30, ], [30, 60], [50, 30], [50, 60]],
            i=lambda *args: [-1, 2000, 4000, 6000],
            angles=lambda *args: [np.pi / 4, -np.pi / 4, np.pi, 0],
            smoke=lambda *args: dict(
                on=True,
                xy=[[0.5, 0.1]],
                inflow=0.01,
                width=60,
                height=6,
                buoyancy=(0, 0.02),
            )
        ),
        test3=dict(
            seed=200,
            initial_conditions_path=root + "/storage/ics/training_3dof/",
            tag='Ba3',
            n_simulations=20,
            positions=lambda i: ((randomGenerator.rand(2) * 40 + 20).tolist(),),
            help_i=lambda *args: [-1],
            i=lambda *args: [-1],
            smoke=lambda *args: dict(
                on=False,
            ),
            n_steps=lambda *args: 1001,
            angles=lambda *args: (randomGenerator.rand(1) * 2 * 3.14159 - 3.14159).tolist()
        ),
        test4=dict(
            initial_conditions_path=root + "/storage/ics/In3/",
            n_simulations=1,
            tag='In3_OneSimulation',
            positions=lambda i: [[75, 40], [75, 70], [120, 40], [120, 70]],
            i=lambda *args: [-1, 2000, 4000, 6000],
            smoke=lambda *args: dict(
                on=False,
            ),
            n_steps=lambda *args: 8001,
            export_stride=lambda *args: 20,
            angles=lambda *args: [1.047197, -1.047197, 2.09439, -1.57079]
        ),
        test5=dict(
            initial_conditions_path=root + "/storage/ics/In3/",
            n_simulations=1,
            tag='InBuo3_OneSimulation',
            positions=lambda i: [[75, 40], [75, 70], [120, 40], [120, 70]],
            i=lambda *args: [-1, 2000, 4000, 6000],
            smoke=lambda *args: dict(
                on=True,
                xy=[[0.50, 0.2]],
                inflow=0.01,
                width=150,
                height=6,
                buoyancy=(0, 0.04),
            ),
            n_steps=lambda *args: 8001,
            export_stride=lambda *args: 20,
            angles=lambda *args: [1.047197, -1.047197, 2.09439, -1.57079]
        ),
        test6=dict(
            initial_conditions_path=root + "/storage/ics/InBuoAg3/",
            n_simulations=1,
            tag='InBuoAg3_OneSimulation',
            positions=lambda i: [[75, 40], [75, 70], [120, 40], [120, 70]],
            i=lambda *args: [-1, 2000, 4000, 6000],
            smoke=lambda *args: dict(
                on=True,
                xy=[[0.50, 0.2]],
                inflow=0.01,
                width=150,
                height=6,
                buoyancy=(0, 0.04),
            ),
            n_steps=lambda *args: 8001,
            export_stride=lambda *args: 20,
            angles=lambda *args: [1.047197, -1.047197, 2.09439, -1.57079]
        ),
        test7=dict(
            initial_conditions_path=root + "/storage/ics/In3/",
            n_simulations=5,
            tag='In3',
            positions=lambda i: [
                [[75, 40], [75, 70], [120, 40], [120, 70]],
                [[75, 70], [120, 70], [75, 40], [120, 40]],
                [[120, 40], [75, 40], [75, 70], [120, 70]],
                [[120, 70], [75, 40], [75, 70], [120, 40]],
                [[75, 40], [120, 70], [75, 70], [120, 40]],
            ][i],
            i=lambda *args: [-1, 2000, 4000, 6000],
            smoke=lambda *args: dict(
                on=False,
                xy=[[0.50, 0.2]],
                inflow=0.01,
                width=150,
                height=6,
                buoyancy=(0, 0.04),
            ),
            n_steps=lambda *args: 8001,
            export_stride=lambda *args: 20,
            angles=lambda i: [
                [1.047197, -1.047197, 2.09439, -1.57079],
                [-1.047197, 1.047197, 2.09439, -1.57079],
                [2.09439, -1.57079, 1.047197, -1.047197],
                [-1.57079, 1.047197, -1.047197, 2.09439, ],
                [1.047197, 2.09439, -1.57079, -1.047197],
            ][i],
        ),
        test8=dict(
            initial_conditions_path=root + "/storage/ics/In3/",
            n_simulations=5,
            tag='InBuo3',
            positions=lambda i: [
                [[75, 40], [75, 70], [120, 40], [120, 70]],
                [[75, 70], [120, 70], [75, 40], [120, 40]],
                [[120, 40], [75, 40], [75, 70], [120, 70]],
                [[120, 70], [75, 40], [75, 70], [120, 40]],
                [[75, 40], [120, 70], [75, 70], [120, 40]],
            ][i],
            i=lambda *args: [-1, 2000, 4000, 6000],
            smoke=lambda *args: dict(
                on=True,
                xy=[[0.50, 0.2]],
                inflow=0.01,
                width=150,
                height=6,
                buoyancy=(0, 0.04),
            ),
            n_steps=lambda *args: 8001,
            export_stride=lambda *args: 20,
            angles=lambda i: [
                [1.047197, -1.047197, 2.09439, -1.57079],
                [-1.047197, 1.047197, 2.09439, -1.57079],
                [2.09439, -1.57079, 1.047197, -1.047197],
                [-1.57079, 1.047197, -1.047197, 2.09439, ],
                [1.047197, 2.09439, -1.57079, -1.047197],
            ][i],
        ),
        test9=dict(
            initial_conditions_path=root + "/storage/ics/InBuoAg3/",
            n_simulations=5,
            tag='InBuoAg3',
            positions=lambda i: [
                [[75, 40], [75, 70], [120, 40], [120, 70]],
                [[75, 70], [120, 70], [75, 40], [120, 40]],
                [[120, 40], [75, 40], [75, 70], [120, 70]],
                [[120, 70], [75, 40], [75, 70], [120, 40]],
                [[75, 40], [120, 70], [75, 70], [120, 40]],
            ][i],
            i=lambda *args: [-1, 2000, 4000, 6000],
            smoke=lambda *args: dict(
                on=True,
                xy=[[0.50, 0.2]],
                inflow=0.01,
                width=150,
                height=6,
                buoyancy=(0, 0.04),
            ),
            n_steps=lambda *args: 8001,
            export_stride=lambda *args: 20,
            angles=lambda i: [
                [1.047197, -1.047197, 2.09439, -1.57079],
                [-1.047197, 1.047197, 2.09439, -1.57079],
                [2.09439, 1.047197, -1.57079, -1.047197],
                [-1.57079, 1.047197, -1.047197, 2.09439, ],
                [1.047197, 2.09439, -1.57079, -1.047197],
            ][i],
        ),
        test10=dict(
            initial_conditions_path=root + "/storage/ics/Buo2/",
            n_simulations=5,
            tag='Buo2',
            help_i=lambda *args: -1,
            export_stride=lambda *args: 20,
            n_steps=lambda *args: 8001,
            positions=lambda i: [
                [[30, 30], [30, 60], [50, 30], [50, 60]],
                [[30, 60], [50, 30], [30, 30], [50, 60]],
                [[50, 30], [50, 60], [30, 60], [30, 30]],
                [[50, 60], [30, 30], [50, 30], [30, 60]],
                [[30, 30], [50, 30], [30, 60], [50, 60]],
            ][i],
            i=lambda *args: [-1, 2000, 4000, 6000],
            angles=lambda *args: [np.pi / 4, -np.pi / 4, np.pi, 0],
            smoke=lambda *args: dict(
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
        randomGenerator.seed(test_attrs.pop('seed', 0))
        export_dict[test_id] = {f"case{i}": dict(positions=test_attrs['positions'](i)) for i in range(test_attrs['n_simulations'])}
        # Add remaining attributes
        print(f"Generating test {test_id}")
        for i, case_key in enumerate(export_dict[test_id]):
            for key, value in test_attrs.items():
                if key in ["initial_conditions_path", "n_simulations", "tag"]: continue  # add this to root of test
                export_dict[test_id][case_key][key] = value(i)
        export_dict[test_id]["initial_conditions_path"] = test_attrs["initial_conditions_path"]
        export_dict[test_id]["n_simulations"] = test_attrs["n_simulations"]
        export_dict[test_id]["tag"] = test_attrs["tag"]

    with open(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json", "w") as f:
        json.dump(export_dict, f, indent="    ")
    # for test_id, test_attrs in export_dict.items():
    #     positions = [test_attrs[f"case{i}"]["positions"] for i in range(test_attrs["n_simulations"])]
    #     positions = np.array(positions)
    #     plt.title(f'Trajectories {test_id}')
    #     plt.plot(*positions.transpose(), '+', linestyle='None')
        # plt.show()
    print("Done")
