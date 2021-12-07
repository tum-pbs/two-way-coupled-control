import os
import json
import wandb
import argparse


def execute(run_path):
    """
    Execute the log_wandb.py script.
    The results will be automatically added to the wandb project
    """
    ordering_key = dict(
        general_error_xy=1,
        general_error_xy_stdd=2,
        stopping_error_xy=3,
        stopping_error_xy_stdd=4,
        force=5,
        force_stdd=6,
    )
    run_name = run_path.split('/')[-2]
    # Load inputs json file
    with open(run_path + '/inputs.json', 'r') as f:
        inp = json.load(f)
    # Load test metrics json file
    tests = os.listdir(run_path + '/tests/')
    os.environ["WANDB_DIR"] = run_path
    os.environ["WANDB_MODE"] = "run"  # Run wandb online so we can sync with previous run
    wandb.init(project="neural_controller_translation_only" if inp["translation_only"] else "neural_controller", name=run_name)
    wandb.config.update(inp)
    wandb.run.name = run_name
    for test in tests:
        try:
            with open(run_path + f'/tests/{test}/metrics.json', 'r') as f:
                metrics = json.load(f)
        except:
            print(f"Did not find metrics for test {test}")
            continue
        test_label = test.split('_')[0]
        metrics_labels = list(metrics.keys())
        metrics_values = list(metrics.values())
        # Sort metrics by amount of values
        metrics_labels, metrics_values = zip(*sorted(zip(metrics_labels, metrics_values), key=lambda metrics: ordering_key[metrics[0]]))
        for i in range(len(metrics_values[0])):
            for label, values in zip(metrics_labels, metrics_values):
                if len(values) > 1:
                    wandb.log({f'{test_label}_{label}': values[i], "t": i})
                elif i == 0:
                    wandb.log({f'{test_label}_{label}': values[i], "t": i})
    # Append run id to inputs file
    with open(run_path + '/inputs.json', 'w') as f:
        inp["id"] = wandb.run.id
        json.dump(inp, f, indent="    ")
    # Finish logging
    # wandb.run.save()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Log results to Weights and Biases")
    parser.add_argument("run_path", help="path to run folder")
    run_path = parser.parse_args().run_path
    execute(run_path)
