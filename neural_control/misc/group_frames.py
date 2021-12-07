import argparse
from natsort import natsorted
import os
import numpy as np


def group_frames(data_dir):
    variables = os.listdir(data_dir)
    variables_iter = iter(variables)
    for variable in variables_iter:
        files = natsorted(os.listdir(data_dir + variable))
        cases = natsorted(list(set([file.split("case")[1][:4] for file in files])))
        for case in cases:
            # print(f"Grouping frames for {variable} case {case}")
            files_to_load = [file for file in files if f"case{case}_" in file]
            if files_to_load == []: continue
            try:
                first_snapshot = np.load(data_dir + variable + "/" + files_to_load[0])
            except ValueError:
                print(f"\n\n   Problem with {variable} case {case}. Skipping \n\n")
                continue
            shape = first_snapshot.shape
            if shape == (): shape = (1,)
            if np.prod(shape) > 20: break  # Skip if too big
            frames = np.zeros((len(files_to_load), *shape))
            for i, file in enumerate(files_to_load):
                frames[i, ...] = np.load(f"{data_dir}/{variable}/{file}")
            np.save(f"{data_dir}/{variable}/{variable}_case{case}.npy", frames)
            # # Erase individual frame files
            # for file in files_to_load:
            #     os.remove(data_dir + variable + "/" + file)


def execute(input_dir):

    data_dir = input_dir + "/data/"
    print(f"Grouping frames of {input_dir}")
    if os.path.exists(data_dir):
        group_frames(data_dir)
        print("Done grouping frames for data")
    else:
        print("No data directory found")
    print("Now grouping frames for tests")
    tests_dir_root = input_dir + "/tests/"
    if os.path.exists(tests_dir_root):
        tests_folders = os.listdir(tests_dir_root)
        for test_folder in tests_folders:
            group_frames(tests_dir_root + test_folder + "/data/")
    else: print(f"Test dir not found in {input_dir}")
    print("Done grouping frames for tests")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Concatenate frames')
    argparser.add_argument('input_dir', type=str, help='Input directory')
    input_dir = argparser.parse_args().input_dir
    execute(input_dir)
